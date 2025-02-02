import os
import pickle
import random
from typing import Dict

import numpy as np
from mesa.agent import AgentSet
from mesa.model import Model
from mesa.datacollection import DataCollector
from tqdm import tqdm

from utils.simulationObjects import Config, Individual, Company, ProductGroup


class EconomyStats:
    total_funds = 0
    individual_wealth_gini = 0
    avg_individual_wealth = 0
    sum_individual_wealth = 0
    sum_company_wealth = 0
    num_companies = 0
    unemployment_rate = 0
    bankruptcies_over_time = 0
    new_companies_over_time = 0

    avg_product_quality = 0
    avg_product_price = 0

    # all_company_funds: list[list[float]] = []
    # all_individual_funds: list[list[float]] = []
    # all_product_prices: list[list[float]] = []
    # all_salaries: list[list[float]] = []


class Economy(Model, EconomyStats):
    def __init__(self, config: Config = Config):
        self.config = config

        super().__init__(seed=self.config.SEED)

        self._create_individuals()
        self._create_companies()

        self.num_bankruptcies = 0
        self.num_new_companies = 0

        # Setup data collector to collect from all attributes in the EconomyStats class
        self.datacollector = DataCollector(
            model_reporters={k: k for k in EconomyStats.__dict__.keys() if not k.startswith('_')}
        )

        # Collect initial stats
        self.refresh_stats()
        self.datacollector.collect(self)

    def _create_individuals(self):
        for _ in range(self.config.NUM_INDIVIDUAL):
            talent = np.random.normal(self.config.TALENT_MEAN, self.config.TALENT_STD)
            initial_funds = np.random.exponential(self.config.INITIAL_FUNDS_INDIVIDUAL)
            Individual(model=self, talent=talent, initial_funds=initial_funds)

    def _create_companies(self):
        # Get all the Individual agents
        unemployed_individuals = list(self.agents_by_type[Individual])
        if not unemployed_individuals:
            raise ValueError("No individuals in the model, run _create_individuals() first.")

        for _ in range(self.config.NUM_COMPANY):
            owner = random.choice(unemployed_individuals)
            unemployed_individuals.remove(owner)
            initial_funds = np.random.exponential(self.config.INITIAL_FUNDS_COMPANY)
            company = Company(model=self, owner=owner, initial_funds=initial_funds)

            # Hire initial employees
            num_initial_employees = min(company.max_employees, len(unemployed_individuals), random.randint(1, 10))
            potential_employees = sorted(list(unemployed_individuals), key=lambda x: x.talent, reverse=True)[
                                  :int(len(unemployed_individuals) * 0.5)]

            if potential_employees:
                initial_employees = random.sample(potential_employees,
                                                  min(num_initial_employees, len(potential_employees)))
                for employee in initial_employees:
                    initial_salary = 50 + employee.talent * 0.5
                    if company.hire_employee(employee, initial_salary):
                        unemployed_individuals.remove(employee)

    def get_all_products(self) -> ProductGroup:
        return ProductGroup(self.agents_by_type[Company].get('product'))

    def get_all_unemployed(self):
        return self.agents_by_type[Individual].select(lambda i: i.employer is None)

    def save_state(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_state(filename: str) -> 'Economy':
        with open(filename, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def calculate_gini(agents: AgentSet) -> float:
        if (n := len(agents)) == 0:
            return 0
        sorted_wealths = sorted(a.funds for a in agents)
        index = np.arange(1, n + 1)
        return ((2 * index - n - 1) * sorted_wealths).sum() / (n * sum(sorted_wealths))

    def refresh_stats(self):
        set_individuals = self.agents_by_type[Individual]
        set_company = self.agents_by_type[Company]

        all_funds_individual = [i.funds for i in set_individuals]

        self.individual_wealth_gini = self.calculate_gini(set_individuals)
        self.sum_individual_wealth = np.sum(all_funds_individual)
        self.sum_company_wealth = np.sum([c.funds for c in set_company])
        self.num_companies = len(set_company)
        self.unemployment_rate = 1 - len([i for i in set_individuals if i.employer]) / len(set_individuals)
        self.bankruptcies_over_time = self.num_bankruptcies  # Track bankruptcies
        self.new_companies_over_time = self.num_new_companies  # Track new companies

        self.total_funds = self.sum_individual_wealth + self.sum_company_wealth

        # to_low_precision = lambda x: float(np.float32(x))
        # self.all_company_funds = [to_low_precision(c.funds) for c in set_company]
        # self.all_individual_funds = [to_low_precision(i.funds) for i in set_individuals]
        # self.all_product_prices = [to_low_precision(c.product.price) for c in set_company]
        # self.all_salaries = [to_low_precision(e.salary) for e in set_individuals if e.employer]

        if set_company:
            self.avg_product_quality = np.mean([c.product.quality for c in set_company])
            self.avg_product_price = np.mean([c.product.price for c in set_company])
        else:
            self.avg_product_quality = 0
            self.avg_product_price = 0


    def step(self):
        set_individual = self.agents_by_type[Individual]
        set_company = self.agents_by_type[Company]

        # Update company product prices and quality and reset revenue
        set_company.set('revenue', 0)
        set_company.do(Company.update_product_attributes)

        # Individuals purchase product
        all_products = self.get_all_products()
        set_individual.do(Individual.purchase_product, products=all_products)

        # Company pays owner dividends, check for bankruptcy, pays employees salary
        set_company.do(Company.do_finances)

        # Individuals find jobs
        self.get_all_unemployed().shuffle_do(Individual.find_job, companies=list(set_company))

        # Start new companies
        set_individual.select(lambda i: i.funds > self.config.MIN_WEALTH_FOR_STARTUP).do(Individual.start_new_company)
        set_company = self.agents_by_type[Company]

        # Adjust workforce for companies
        unemployed = list(self.get_all_unemployed())
        set_company.shuffle_do(Company.adjust_workforce, unemployed)

        # Collect statistics
        self.refresh_stats()
        self.datacollector.collect(self)


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
        economy = Economy(Config(
            NUM_INDIVIDUAL=num_individuals,
            NUM_COMPANY=num_companies,
        ))
    for _ in tqdm(range(num_steps - economy.steps), desc="Simulating economy"):
        economy.step()
        # economy.all_companies[0].print_statistics()
        # economy.stats.save_stats("simulation_stats.pkl")

        # Save simulation state
        economy.save_state("economy_simulation.pkl")
    return economy


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
        num_steps=1000,
        state_pickle_path="economy_simulation.pkl",
        resume_state=False,
    )

    # Load simulation state (optional)
    # economy = Economy.load_state("economy_simulation.pkl")

    # Plot results
    # plot_results(economy, save_path="economy_simulation_results.png")
    # print_summary(economy)
