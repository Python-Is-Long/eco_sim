import os
import pickle
import random
from typing import List, Dict

import numpy as np
from tqdm import tqdm

from mesa.model import Model

from utils.simulationObjects import Config, Individual, Company, Product, ProductGroup


class Economy(Model):
    def __init__(self, config: Config = Config):
        self.config = config

        super().__init__(seed=self.config.SEED)

        self._create_individuals()
        self._create_companies()
        self.stats = EconomyStats()

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
    @property
    def individuals(self):
        return list(self.agents_by_type[Individual])

    @property
    def companies(self):
        return list(self.agents_by_type[Company])

    def get_all_products(self) -> ProductGroup:
        return ProductGroup(self.agents_by_type[Company].get('product'))

    def get_all_unemployed(self):
        return self.agents_by_type[Individual].select(lambda i:i.employer is None)

    def save_state(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_state(filename: str) -> 'Economy':
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def step(self):
        self.stats.step += 1
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
        set_individual.select(lambda i:i.funds > self.config.MIN_WEALTH_FOR_STARTUP).do(Individual.start_new_company)
        set_company = self.agents_by_type[Company]

        # Adjust workforce for companies
        unemployed = list(self.get_all_unemployed())
        set_company.shuffle_do(Company.adjust_workforce, unemployed)

        # Collect statistics
        # TODO: Use the data collection system provided by mesa
        self.stats.collect_statistics(self)


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
                ['all_company_funds', 'all_individual_funds', 'all_product_prices', 'all_salaries']}

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

        if economy.companies:
            self.avg_product_quality.append(np.mean([c.product.quality for c in economy.companies]))
            self.avg_product_price.append(np.mean([c.product.price for c in economy.companies]))
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
        economy = Economy(Config(
            NUM_INDIVIDUAL=num_individuals,
            NUM_COMPANY=num_companies,
        ))
    for _ in tqdm(range(num_steps - economy.stats.step), desc="Simulating economy"):
        economy.step()
        # economy.all_companies[0].print_statistics()
        economy.stats.save_stats("simulation_stats.pkl")

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