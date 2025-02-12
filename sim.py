import os
import pickle
import random
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json

from utils.simulationObjects import Config, Individual, Company, Product, ProductGroup
from utils.simulationObjects import IndividualReports, CompanyReports
from utils.database import DatabaseInfo, SimDatabase


class Economy:
    def __init__(self, db_info: DatabaseInfo, config: Config = Config()):
        self.config = config

        self.individuals = self._create_individuals(self.config.NUM_INDIVIDUAL)
        self.companies = self._create_companies(self.config.NUM_COMPANY)
        self.all_companies = self.companies.copy()
        self.stats = EconomyStats()
        self.report_types = [IndividualReports, CompanyReports]
        self.db = SimDatabase(db_info, self.report_types)
        self.db.create_database('ECOSIM')

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

    def get_all_products(self) -> ProductGroup:
        return ProductGroup([company.product for company in self.companies])

    def simulate_step(self):
        self.stats.step += 1
        # Update company product prices and quality and reset revenue
        for company in self.companies:
            company.remove_raw_material() # see if remove raw_material at this step
            company.update_product_attributes(population=len(self.individuals), company_count=len(self.companies))
            company.revenue = 0

        # see if able to find a raw_material at this step
        if len(self.companies) > 1:
            for company in self.companies:
                all_raw_materials = [c.product for c in self.companies if c is not company]
                company.find_raw_material(all_raw_materials)

        all_products = self.get_all_products()

        # Individuals spend money
        for individual in self.individuals:
            chosen_product = individual.decide_purchase(all_products)
            if chosen_product:
                individual.make_purchase(self.companies, chosen_product)
                individual.expenses = chosen_product.price
                # print(f'{individual.name} is buying product {chosen_product.name} from company {chosen_product.company.name} for {chosen_product.price}')

        for company in self.companies:
            # Pay dividends from profit to owner
            company.transfer_funds_to(company.owner, company.dividend)

            # Check for bankruptcy
            if company.check_bankruptcy():
                self.companies.remove(company)
                self.stats.num_bankruptcies += 1
                continue
            # Pays employees salary
            for employee in company.employees:
                company.transfer_funds_to(employee, employee.salary)
            if company.funds < 0:
                print('negative funds:', company.funds)

        # Individuals find jobs
        for individual in self.individuals:
            if individual.employer is None:
                individual.find_job(self.companies, individual.unemployed_state)

            if individual.employer is None and individual.unemployed_state <5:
                individual.unemployed_state += 1

        # TODO: Handles case where len(self.companies) is 0
        market_potential = np.log10(len(self.individuals)) / len(self.companies)
        # Start new companies
        for individual in self.individuals:
            be_entrepreneur = Individual.choose_niche(individual, niches=self.config.POSSIBLE_MARKETS)
            # TODO: Instead of random chance of starting a new company, consider the current market demands
            if individual.funds > self.config.MIN_WEALTH_FOR_STARTUP and random.random() < market_potential and be_entrepreneur:
                self.start_new_company(individual)

        # Adjust workforce for companies
        for company in self.companies:
            self.adjust_workforce(company)

        # Insert reports into database
        self.db.insert_reports([i.report() for i in self.individuals])
        self.db.insert_reports([c.report() for c in self.companies])

        # Collect statistics
        self.stats.collect_statistics(self)

    def start_new_company(self, individual: Individual):
        if individual.funds > self.config.MIN_WEALTH_FOR_STARTUP:
            initial_funds = individual.funds * self.config.STARTUP_COST_FACTOR
            new_company = Company(self, individual)
            individual.transfer_funds_to(new_company, initial_funds)
            self.companies.append(new_company)
            self.all_companies.append(new_company)
            self.stats.num_new_companies += 1  # Increment new company counter

    def adjust_workforce(self, company: Company):
        if company.revenue > company.costs * self.config.PROFIT_MARGIN_FOR_HIRING:
            # Hire new employees
            potential_employees = [ind for ind in self.individuals if ind.employer is None]
            if potential_employees:
                new_employee = max(potential_employees, key=lambda x: x.talent)
                company.hire_employee(new_employee, new_employee.talent * self.config.SALARY_FACTOR)
        elif company.revenue < company.costs:
            # Fire employees
            if company.employees:
                employee_to_fire = random.choice(company.employees)
                company.fire_employee(employee_to_fire)
            # TODO: add a bankruptcy index every time when there's no worker to fire

    def save_state(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_state(filename: str) -> 'Economy':
        with open(filename, 'rb') as f:
            return pickle.load(f)


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

    def calculate_gini(self, wealths: List[float]) -> float:
        sorted_wealths = sorted(wealths)
        n = len(sorted_wealths)
        if n == 0:
            return 0
        index = np.arange(1, n + 1)
        return ((2 * index - n - 1) * sorted_wealths).sum() / (n * sum(sorted_wealths))

    def save_stats(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    def collect_statistics(self, economy: Economy):
        individual_wealths = [ind.funds for ind in economy.individuals]
        company_wealths = [comp.funds for comp in economy.companies]
        employed = len([ind for ind in economy.individuals if ind.employer])

        self.individual_wealth_gini.append(self.calculate_gini(individual_wealths))
        self.avg_individual_wealth.append(np.mean(individual_wealths))
        self.sum_individual_wealth.append(np.sum(individual_wealths))
        self.sum_company_wealth.append(np.sum(company_wealths))
        self.num_companies.append(len(economy.companies))
        self.unemployment_rate.append(1 - employed / len(economy.individuals))
        self.bankruptcies_over_time.append(self.num_bankruptcies)  # Track bankruptcies
        self.new_companies_over_time.append(self.num_new_companies)  # Track new companies

        self.total_money = round(self.sum_individual_wealth[-1] + self.sum_company_wealth[-1])

        to_low_precision = lambda x: float(np.float32(x))
        self.all_company_funds.append([to_low_precision(c.funds) for c in economy.companies])
        self.all_individual_funds.append([to_low_precision(i.funds) for i in economy.individuals])
        self.all_product_prices.append([to_low_precision(c.product.price) for c in economy.companies])
        self.all_salaries.append([to_low_precision(e.salary) for e in economy.individuals if e.employer])
        self.all_employee_counts.append([len(c.employees) for c in economy.companies])

        if economy.companies:
            self.avg_product_quality.append(np.mean([c.product.quality for c in economy.companies]))
            self.avg_product_price.append(np.mean([c.product.price for c in economy.companies]))
            self.avg_company_raw_materials.append(np.mean([len(c.raw_materials) for c in economy.companies]))
            self.avg_company_employees.append(np.mean([len(c.employees) for c in economy.companies]))
        else:
            self.avg_product_quality.append(0)
            self.avg_product_price.append(0)


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
        economy = Economy(DatabaseInfo(**db_info), config=Config(
            NUM_INDIVIDUAL=num_individuals,
            NUM_COMPANY=num_companies,
        ))
    for _ in tqdm(range(num_steps - economy.stats.step), desc="Simulating economy"):
        economy.simulate_step()
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