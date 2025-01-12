import os
import pickle
import random
import uuid
from dataclasses import dataclass
from typing import List, Optional, Union, Any, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils.genericObjects import NamedObject, FundsObject


# Configuration
class Config:
    # Individual settings
    INITIAL_FUNDS_INDIVIDUAL = 1000  # Initial funds for individuals
    FUNDS_PRECISION = np.float64
    TALENT_MEAN = 100  # Mean talent (IQ-like)
    TALENT_STD = 15  # Standard deviation of talent

    # Company settings
    INITIAL_FUNDS_COMPANY = 5000000  # Increased initial funds
    MIN_COMPANY_SIZE = 5  # Minimum initial company size
    MAX_COMPANY_SIZE = 20  # Maximum initial company size
    SALARY_FACTOR = 100  # Salary = talent * SALARY_FACTOR
    DIVIDEND_RATE = 0.1  # 10% of profit paid as dividends
    PROFIT_MARGIN_FOR_HIRING = 1.5  # Higher margin for hiring
    BANKRUPTCY_THRESHOLD = 1000  # Companies can go slightly negative before bankruptcy

    # Spending settings
    SPENDING_PROBABILITY_FACTOR = 5000  # Wealth factor for spending probability

    # Entrepreneurship settings
    STARTUP_COST_FACTOR = 0.5  # Fraction of wealth used to start a company
    MIN_WEALTH_FOR_STARTUP = 10000  # Minimum wealth to start a company


@dataclass
class Product:
    quality: float
    price: float
    company: 'Company'
    name: str

class Individual(NamedObject, FundsObject, funds_precision=Config.FUNDS_PRECISION):
    def __init__(self, talent: float, initial_funds: float):
        self.set_funds(initial_funds)
        self.talent = talent
        self.employer: Optional[Company] = None
        self.salary = self.funds_precision(0)

    def make_purchase(self, product: Product):
        self.transfer_funds_to(product.company, product.price)
        product.company.revenue += product.price

    @staticmethod
    def score_product(product: Product, wealth_factor) -> float:
        quality_weight = 0.5 + 0.5 * wealth_factor
        price_weight = 1.5 - 1.5 * wealth_factor
        return quality_weight * product.quality - price_weight * product.price

    def decide_purchase(self, products: List[Product]) -> Optional[Product]:
        if not products:
            return None

        # Wealthy individuals care more about quality than price
        wealth_factor = np.tanh(self.funds / Config.SPENDING_PROBABILITY_FACTOR)

        scored_products = [(self.score_product(p, wealth_factor), p) for p in products]
        if not scored_products:
            return None
        # Filter out all unaffordable products
        scored_products = [(score, product) for score, product in scored_products if self.can_afford(product.price)]
        if not scored_products:
            return None

        return max(scored_products, key=lambda x: x[0])[1]

    def find_job(self, companies: List['Company']):
        if self.employer is None:
            for company in companies:
                if company.hire_employee(self, self.talent * Config.SALARY_FACTOR):
                    break


class Company(NamedObject, FundsObject, funds_precision=Config.FUNDS_PRECISION):
    def __init__(self, owner: Individual, initial_funds: float=0):
        self.set_funds(initial_funds)
        self.owner = owner
        
        self.employees: List[Individual] = []
        self.product = Product(quality=1, price=1, company=self, name=uuid.uuid4())
        self.revenue = self.funds_precision(0)
        self.suppliers: List[Company] = []
        self.max_employees = random.randint(Config.MIN_COMPANY_SIZE, Config.MAX_COMPANY_SIZE)
        self.bankruptcy = False

    @property
    def total_salary(self):
        return sum(emp.salary for emp in self.employees)

    @property
    def total_material_cost(self):
        return sum(supplier.product.price for supplier in self.suppliers)

    @property
    def costs(self):
        return self.total_salary + self.total_material_cost

    @property
    def profit(self):
        return self.revenue - self.costs

    @property
    def dividend(self):
        return self.profit * Config.DIVIDEND_RATE if self.profit > 0 else 0

    def calculate_product_quality(self) -> float:
        if not self.employees:
            return 0.0

        # Base quality from employees with diminishing returns
        employee_contribution = sum(emp.talent for emp in self.employees)
        diminishing_factor = np.log(len(self.employees) + 1)
        base_quality = employee_contribution / diminishing_factor

        # Additional quality from suppliers
        supplier_quality = sum(supplier.product.quality for supplier in self.suppliers)
        return base_quality + supplier_quality * 0.5

    def update_product_attributes(self):
        # Update product quality and price
        self.product.quality = max(1, self.calculate_product_quality())  # Ensure quality >= 1
        self.product.price = max(1, self.costs * 1.2)  # Ensure price >= 1


    def hire_employee(self, candidate: Individual, salary: float) -> bool:
        if self.funds >= salary and len(self.employees) < self.max_employees:
            candidate.employer = self
            candidate.salary = salary
            self.employees.append(candidate)
            return True
        return False

    def fire_employee(self, employee: Individual):
        if employee in self.employees:
            self.employees.remove(employee)
            employee.employer = None
            employee.salary = 0.0

    def check_bankruptcy(self) -> bool:
        if self.funds < self.costs:
            # Fire all employees
            for employee in self.employees:
                self.fire_employee(employee)
            # The owner runs with leftover company funds
            self.transfer_funds_to(self.owner, self.funds)
            self.bankruptcy = True
            return True
        return False
    

    def print_statistics(self):
        stats = (
            f"UUID: {self.name}, "
            f"Owner: {self.owner.name}, Funds: {self.funds:.2f}, "
            f"Employees: {len(self.employees)}/{self.max_employees}, "
            f"Total Salary: {sum(emp.salary for emp in self.employees):.2f}, "
            f"Suppliers: {len(self.suppliers)}, "
            f"Product Quality: {self.product.quality:.2f}, Product Price: {self.product.price:.2f}, "
            f"Costs: {self.costs:.2f}, Revenue: {self.revenue:.2f}, Dividands: {self.dividend:.2f}, "
            f"Bankruptcy: {self.bankruptcy}"
        )
        print(stats)


class Economy:
    def __init__(self, num_individuals: int, num_companies: int):
        self.individuals = self._create_individuals(num_individuals)
        self.companies = self._create_companies(num_companies)
        self.all_companies = self.companies.copy()
        self.stats = EconomyStats()

    def _create_individuals(self, num_individuals: int) -> List[Individual]:
        talents = np.random.normal(Config.TALENT_MEAN, Config.TALENT_STD, num_individuals)
        initial_funds = np.random.exponential(Config.INITIAL_FUNDS_INDIVIDUAL, num_individuals)
        return [Individual(t, f) for t, f in zip(talents, initial_funds)]

    def _create_companies(self, num_companies: int) -> List[Company]:
        companies = []
        available_workers = set(self.individuals)

        for _ in range(num_companies):
            owner = random.choice(list(available_workers))
            available_workers.remove(owner)
            initial_funds = np.random.exponential(Config.INITIAL_FUNDS_COMPANY)
            company = Company(owner, initial_funds)

            # Hire initial employees
            num_initial_employees = min(company.max_employees, len(available_workers), random.randint(1, 10))
            potential_employees = sorted(list(available_workers), key=lambda x: x.talent, reverse=True)[:int(len(available_workers) * 0.5)]

            if potential_employees:
                initial_employees = random.sample(potential_employees, min(num_initial_employees, len(potential_employees)))
                for employee in initial_employees:
                    initial_salary = 50 + employee.talent * 0.5
                    if company.hire_employee(employee, initial_salary):
                        available_workers.remove(employee)

            companies.append(company)

        return companies

    def get_all_products(self):
        return [company.product for company in self.companies]

    def simulate_step(self):
        self.stats.step += 1
        # Update company product prices and quality and reset revenue
        for company in self.companies:
            company.update_product_attributes()
            company.revenue = 0

        # Individuals spend money
        for individual in self.individuals:
            if random.random() < np.tanh(individual.funds / Config.SPENDING_PROBABILITY_FACTOR):
                chosen_product = individual.decide_purchase(self.get_all_products())
                if chosen_product:
                    individual.make_purchase(chosen_product)
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
                individual.find_job(self.companies)

        # Start new companies
        for individual in self.individuals:
            # TODO: Instead of random chance of starting a new company, consider the current market demands
            if individual.funds > Config.MIN_WEALTH_FOR_STARTUP and random.random() < 0.01:
                self.start_new_company(individual)

        # Adjust workforce for companies
        for company in self.companies:
            self.adjust_workforce(company)

        # Collect statistics
        self.stats.collect_statistics(self)

    def start_new_company(self, individual: Individual):
        if individual.funds > Config.MIN_WEALTH_FOR_STARTUP:
            initial_funds = individual.funds * Config.STARTUP_COST_FACTOR
            new_company = Company(individual)
            individual.transfer_funds_to(new_company, initial_funds)
            self.companies.append(new_company)
            self.all_companies.append(new_company)
            self.stats.num_new_companies += 1  # Increment new company counter

    def adjust_workforce(self, company: Company):
        if company.revenue > company.costs * Config.PROFIT_MARGIN_FOR_HIRING:
            # Hire new employees
            potential_employees = [ind for ind in self.individuals if ind.employer is None]
            if potential_employees:
                new_employee = max(potential_employees, key=lambda x: x.talent)
                company.hire_employee(new_employee, new_employee.talent * Config.SALARY_FACTOR)
        elif company.revenue < company.costs:
            # Fire employees
            if company.employees:
                employee_to_fire = random.choice(company.employees)
                company.fire_employee(employee_to_fire)

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

        self.all_company_funds: List[List[float]] = []
        self.all_individual_funds: List[List[float]] = []
        self.all_product_prices: List[List[float]] = []
        self.all_salaries: List[List[float]] = []

    @property
    def dict_scalar_attributes(self) -> Dict:
        return {attr: value for attr, value in self.__dict__.items() if isinstance(value, (int, float))}
    
    @property
    def dict_histogram_attributes(self) -> Dict:
        return {attr: self.__dict__[attr] for attr in ['all_company_funds', 'all_individual_funds', 'all_product_prices', 'all_salaries']}

    @property
    def dict_time_series_attributes(self) -> Dict:
        return {attr: value for attr, value in self.__dict__.items() if isinstance(value, list) and attr not in self.dict_histogram_attributes}

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
        economy = Economy(num_individuals, num_companies)
    for _ in tqdm(range(num_steps-economy.stats.step), desc="Simulating economy"):
        economy.simulate_step()
        # economy.all_companies[0].print_statistics()
        economy.stats.save_stats("simulation_stats.pkl")

        # Save simulation state
        economy.save_state("economy_simulation.pkl")
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
        num_individuals=10000,
        num_companies=50,
        num_steps=1000,
        state_pickle_path="economy_simulation.pkl",
        resume_state=False,
    )

    # Load simulation state (optional)
    # economy = Economy.load_state("economy_simulation.pkl")

    # Plot results
    # plot_results(economy, save_path="economy_simulation_results.png")
    # print_summary(economy)