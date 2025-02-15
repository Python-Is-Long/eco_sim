import os
import pickle
import random
from typing import List, Dict, Optional, Sequence, Callable
from threading import Thread, Event
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json

from utils.simulationObjects import Config, Individual, Company, ProductGroup
from utils.simulationObjects import IndividualReports, CompanyReports
from utils.database import DatabaseInfo, SimDatabase

step_count = {}

def staged_stepping(
    agent,
    stages: Sequence[Callable[..., bool] | Sequence[Callable[..., bool]] | None],
    start_stage_event: Event,
    end_stage_event: Event
):
    global step_count
    step_count[agent] = 0
    current_step = 0
    still_live = True
    while still_live:
        current_step += 1
        for i, stage in enumerate(stages):
            # Wait for the start of the stage
            start_stage_event.wait()
            # Clear the end stage event immediately for the main thread will wait before this stage completes
            end_stage_event.clear()
            step_count[agent] = current_step + i / 10
            # Execute the stage
            if stage is not None:
                if callable(stage):
                    still_live = stage(agent)
                else:
                    for s in stage:
                        still_live = s(agent)
                        if not still_live:
                            break
            # Notify the main thread that the stage is complete
            end_stage_event.set()
            # Terminates the thread if the agent is no longer alive
            if not still_live:
                break


class Economy:
    creating_companies: List
    removing_companies: List[Company]
    individual_stages: Sequence[Callable[..., bool] | None]
    company_stages: Sequence[Callable[..., bool] | None]

    def __init__(self, db_info: Optional[DatabaseInfo] = None, config: Config = Config()):
        self.config = config

        self.individuals = self._create_individuals(self.config.NUM_INDIVIDUAL)
        self.companies = self._create_companies(self.config.NUM_COMPANY)
        self.market_potential = 0
        self.all_companies = self.companies.copy()
        self.stats = EconomyStats()
        self.report_types = [IndividualReports, CompanyReports]
        self.creating_companies = []
        self.removing_companies = []
        self.start_stage_event = Event()
        self.end_stage_events = []
        self.threads = []

        self.db_connection = True if db_info else False
        if db_info:
            self.db = SimDatabase(db_info, self.report_types)
            self.db.create_database('ECOSIM')

        # Economy methods that runs before each stage
        self.before_stage = [
            [self.collect_statistics, self.reports],
            None,
            None,
            self.handle_company_removal,
            self.handle_company_creation,
        ]
        self.individual_stages = [
            None,
            Individual.purchase_product,  # Individuals spend money
            None,
            Individual.find_opportunities,  # Individuals find jobs or create new company
            None,
        ]
        self.company_stages = [
            Company.update_product, # Update company product prices and quality and find new materials
            None,
            Company.do_finance,  # Company checks for bankruptcy and pays dividends then pays salaries
            None,
            Company.adjust_workforce,  # Adjust workforce for companies
        ]

        assert len(self.before_stage) == len(self.individual_stages) == len(self.company_stages), \
            'All stages must be the same length!'

        [self.create_thread(i) for i in self.individuals]
        [self.create_thread(c) for c in self.companies]

    def _create_individuals(self, num_individuals: int) -> List[Individual]:
        talents = np.random.normal(self.config.TALENT_MEAN, self.config.TALENT_STD, num_individuals)
        initial_funds = np.random.exponential(self.config.INITIAL_FUNDS_INDIVIDUAL, num_individuals)
        risk_tolerance = [round(random.uniform(0.5, 2.0), 2) for _ in range(num_individuals)]
        skills = [set(random.choices(self.config.POSSIBLE_MARKETS, k=self.config.MAX_SKILLS)) for _ in range(num_individuals)]
        return [Individual(self, t, f, skills=s, risk_tolerance=r, configuration=self.config) for t, f, s, r in zip(talents, initial_funds, skills, risk_tolerance)]

    def _create_companies(self, num_companies: int) -> List[Company]:
        companies = []
        available_workers = set(self.individuals)

        for _ in range(num_companies):
            owner = random.choice(list(available_workers))
            available_workers.remove(owner)
            initial_funds = np.random.exponential(self.config.INITIAL_FUNDS_COMPANY)
            company = Company(self, owner, initial_funds)

            # Hire initial employees
            num_initial_employees = min(company.max_employees, len(available_workers), random.randint(1, 10))
            potential_employees = sorted(list(available_workers), key=lambda x: x.talent, reverse=True)[
                                  :int(len(available_workers) * 0.5)]

            if potential_employees:
                initial_employees = random.sample(potential_employees,
                                                  min(num_initial_employees, len(potential_employees)))
                for employee in initial_employees:
                    initial_salary = 50 + employee.talent * 0.5
                    if company.hire_employee(employee, initial_salary):
                        available_workers.remove(employee)

            companies.append(company)
        return companies

    def create_thread(self, agent: Individual | Company):
        end_stage_event = Event()
        self.end_stage_events.append(end_stage_event)
        stages = self.company_stages if isinstance(agent, Company) else self.individual_stages
        thread = Thread(
            target=staged_stepping,
            args=(agent, stages, self.start_stage_event, end_stage_event),
            daemon=True,
        )
        self.threads.append(thread)
        thread.start()

    def get_all_products(self) -> ProductGroup:
        return ProductGroup([company.product for company in self.companies])

    def get_all_unemployed(self):
        return [i for i in self.individuals if i.employer is None]

    def simulate_step(self):
        self.stats.step += 1
        # Collect statistics
        self.collect_statistics()

        for company in self.companies:
            company.update_product()

        # Individuals spend money
        for individual in self.individuals:
            individual.purchase_product()

        # Company checks for bankruptcy and pays dividends then pays salaries
        for company in self.companies:
            company.do_finance()
        self.handle_company_removal()

        # Individuals find jobs or create new company
        self.update_market_potential()
        for individual in self.individuals:
            individual.find_opportunities()
        self.handle_company_creation()

        # Adjust workforce for companies
        for company in self.companies:
            company.adjust_workforce()

        # Insert reports into database if there is a connection
        self.reports()

    def simulate_step_threaded(self):
        for e in self.before_stage:
            # Execute the code before a stage starts
            if e is not None:
                if callable(e):
                    e()
                else:
                    [i() for i in e if i]
            # Start the stage
            self.start_stage_event.set()
            time.sleep(0.001)
            self.start_stage_event.clear()
            [e.wait() for e in self.end_stage_events]

            # Check if all threads are on the same step
            if len(set(step_count.values())) == 1:
                print("Threads are not on the same step: {step_count}")
                print(step_count)

    def reports(self):
        if self.db_connection:
            self.db.insert_reports([i.report() for i in self.individuals])
            self.db.insert_reports([c.report() for c in self.companies])

    def handle_company_creation(self):
        for c in self.creating_companies:
            new_company = Company(self, c.owner)
            c.owner.transfer_funds_to(new_company, c.initial_funds)
            self.companies.append(new_company)
            self.stats.num_new_companies += 1
        self.creating_companies = []

    def handle_company_removal(self):
        [self.companies.remove(c) for c in self.removing_companies]
        self.removing_companies = []

    def update_market_potential(self):
        # TODO: Handles case where len(self.companies) is 0
        self.market_potential = np.log10(len(self.individuals)) / len(self.companies)

    def save_state(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_state(filename: str) -> 'Economy':
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def collect_statistics(self):
        individual_wealths = [ind.funds for ind in self.individuals]
        company_wealths = [comp.funds for comp in self.companies]
        employed = len([ind for ind in self.individuals if ind.employer])

        self.stats.individual_wealth_gini.append(self.stats.calculate_gini(individual_wealths))
        self.stats.avg_individual_wealth.append(np.mean(individual_wealths))
        self.stats.sum_individual_wealth.append(np.sum(individual_wealths))
        self.stats.sum_company_wealth.append(np.sum(company_wealths))
        self.stats.num_companies.append(len(self.companies))
        self.stats.unemployment_rate.append(1 - employed / len(self.individuals))
        self.stats.bankruptcies_over_time.append(self.stats.num_bankruptcies)  # Track bankruptcies
        self.stats.new_companies_over_time.append(self.stats.num_new_companies)  # Track new companies

        self.stats.total_money = round(self.stats.sum_individual_wealth[-1] + self.stats.sum_company_wealth[-1])

        to_low_precision = lambda x: float(np.float32(x))
        self.stats.all_company_funds.append([to_low_precision(c.funds) for c in self.companies])
        self.stats.all_individual_funds.append([to_low_precision(i.funds) for i in self.individuals])
        self.stats.all_product_prices.append([to_low_precision(c.product.price) for c in self.companies])
        self.stats.all_salaries.append([to_low_precision(e.salary) for e in self.individuals if e.employer])
        self.stats.all_employee_counts.append([len(c.employees) for c in self.companies])

        if self.companies:
            self.stats.avg_product_quality.append(np.mean([c.product.quality for c in self.companies]))
            self.stats.avg_product_price.append(np.mean([c.product.price for c in self.companies]))
            self.stats.avg_company_raw_materials.append(np.mean([len(c.raw_materials) for c in self.companies]))
            self.stats.avg_company_employees.append(np.mean([len(c.employees) for c in self.companies]))
        else:
            self.stats.avg_product_quality.append(0)
            self.stats.avg_product_price.append(0)


class EconomyStats:
    def __init__(self):
        self.step = 0
        self.total_money = 0
        self.num_bankruptcies = 0
        self.num_new_companies = 0

        self.individual_wealth_gini = []
        self.avg_individual_wealth = []
        self.sum_individual_wealth = []
        self.sum_company_wealth = []
        self.num_companies = []
        self.unemployment_rate = []
        self.avg_product_quality = []
        self.avg_product_price = []
        self.bankruptcies_over_time = []
        self.new_companies_over_time = []
        self.avg_company_raw_materials = []
        self.avg_company_employees = []
        self.all_employee_counts = []

        self.all_company_funds: List[List[float]] = []
        self.all_individual_funds: List[List[float]] = []
        self.all_product_prices: List[List[float]] = []
        self.all_salaries: List[List[float]] = []

    @property
    def dict_scalar_attributes(self) -> Dict:
        return {attr: value for attr, value in self.__dict__.items() if isinstance(value, (int, float))}

    @property
    def dict_histogram_attributes(self) -> Dict:
        return {attr: self.__dict__[attr] for attr in
                ['all_company_funds', 'all_individual_funds', 'all_product_prices', 'all_salaries', 'all_employee_counts']}

    @property
    def dict_time_series_attributes(self) -> Dict:
        return {attr: value for attr, value in self.__dict__.items() if
                isinstance(value, list) and attr not in self.dict_histogram_attributes}

    @staticmethod
    def calculate_gini(wealths: List[float]) -> float:
        sorted_wealths = sorted(wealths)
        n = len(sorted_wealths)
        if n == 0:
            return 0
        index = np.arange(1, n + 1)
        return ((2 * index - n - 1) * sorted_wealths).sum() / (n * sum(sorted_wealths))

    def save_stats(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)


def run_simulation(
        num_individuals: int = 1000,
        num_companies: int = 50,
        num_steps: int = 100,
        state_pickle_path: str = "economy_simulation.pkl",
        resume_state: bool = False,
) -> Economy:
    # Load simulation state (optional)
    if resume_state and os.path.exists(state_pickle_path):
        economy = Economy.load_state("economy_simulation.pkl")
        print(f"Resuming simulation from saved state: {state_pickle_path}")
    else:
        with open('db_info.json', 'r') as f:
            db_info = json.load(f)
        config = Config(
            NUM_INDIVIDUAL=num_individuals,
            NUM_COMPANY=num_companies,
        )
        # Note: Economy with database connection will not be picklable
        economy = Economy(
            # DatabaseInfo(**db_info),
            config=config,
        )
    for _ in tqdm(range(num_steps - economy.stats.step), desc="Simulating economy"):
        # economy.simulate_step()
        economy.simulate_step_threaded()
        # economy.all_companies[0].print_statistics()
        economy.stats.save_stats("simulation_stats.pkl")

        # Save simulation state
        # economy.save_state("economy_simulation.pkl")
    return economy


# def plot_results(
#         economy: Economy,
#         save_path: Optional[str] = None,
# ):
#     n_rows = 5

#     fig, axes = plt.subplots(n_rows, 2, figsize=(15, 4*n_rows))
#     fig.suptitle('Economic Simulation Results')

#     axes[0, 0].plot(economy.stats.individual_wealth_gini)
#     axes[0, 0].set_title('Wealth Inequality (Gini)')
#     axes[0, 0].set_ylabel('Gini Coefficient')

#     axes[0, 1].plot(economy.stats.avg_individual_wealth)
#     axes[0, 1].set_title('Average Wealth')
#     axes[0, 1].set_ylabel('Wealth')

#     axes[1, 0].plot(economy.stats.num_companies)
#     axes[1, 0].set_title('Number of Companies')
#     axes[1, 0].set_ylabel('Count')

#     axes[1, 1].plot(economy.stats.unemployment_rate)
#     axes[1, 1].set_title('Unemployment Rate')
#     axes[1, 1].set_ylabel('Rate')

#     axes[2, 0].plot(economy.stats.avg_product_quality)
#     axes[2, 0].set_title('Average Product Quality')
#     axes[2, 0].set_ylabel('Quality')

#     axes[2, 1].plot(economy.stats.avg_product_price)
#     axes[2, 1].set_title('Average Product Price')
#     axes[2, 1].set_ylabel('Price')

#     axes[3, 0].plot(economy.stats.bankruptcies_over_time, label='Bankruptcies')
#     axes[3, 0].set_title('Number of Bankruptcies Over Time')
#     axes[3, 0].set_ylabel('Count')

#     axes[3, 1].plot(economy.stats.new_companies_over_time, label='New Companies')
#     axes[3, 1].set_title('Number of New Companies Over Time')
#     axes[3, 1].set_ylabel('Count')

#     axes[4, 0].plot(economy.stats.total_money, label='Total money')
#     axes[4, 0].set_title('Total money')
#     axes[4, 0].set_ylabel('Money')

#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path)

#     plt.show()


def print_summary(economy: Economy):
    print("\nFinal Economic Statistics:")
    print(f"Number of individuals: {len(economy.individuals)}")
    print(f"Number of companies: {len(economy.companies)}")
    print(f"Average wealth: {economy.stats.avg_individual_wealth[-1]:.2f}")
    print(f"Wealth inequality (Gini): {economy.stats.individual_wealth_gini[-1]:.2f}")
    print(f"Unemployment rate: {economy.stats.unemployment_rate[-1]:.2%}")
    print(f"Total bankruptcies: {economy.stats.num_bankruptcies}")
    print(f"Total new companies: {economy.stats.num_new_companies}")

    wealth_percentiles = np.percentile([ind.funds for ind in economy.individuals], [25, 50, 75, 90, 99])
    print("\nWealth Distribution:")
    print(f"25th percentile: {wealth_percentiles[0]:.2f}")
    print(f"Median: {wealth_percentiles[1]:.2f}")
    print(f"75th percentile: {wealth_percentiles[2]:.2f}")
    print(f"90th percentile: {wealth_percentiles[3]:.2f}")
    print(f"99th percentile: {wealth_percentiles[4]:.2f}")


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    # # Run simulation
    economy = run_simulation(
        num_individuals=100,
        num_companies=5,
        num_steps=100,
        state_pickle_path="economy_simulation.pkl",
        resume_state=False,
    )

    # Load simulation state (optional)
    # economy = Economy.load_state("economy_simulation.pkl")

    # Plot results
    # plot_results(economy, save_path="economy_simulation_results.png")
    # print_summary(economy)