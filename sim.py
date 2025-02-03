import os
import pickle
import random
import uuid
from dataclasses import dataclass
from random import choices, uniform
from typing import List, Optional, Union, Any, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils.genericObjects import NamedObject, FundsObject


# Configuration
class Config:
    # Individual settings
    INITIAL_FUNDS_INDIVIDUAL = 1000  # Initial funds for individuals
    FUNDS_PRECISION = np.float64 # Object used to store funds (int or float like)
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

    # Entrepreneurship settings
    MAX_SKILLS = 3
    STARTUP_COST_FACTOR = 0.5  # Fraction of wealth used to start a company
    MIN_WEALTH_FOR_STARTUP = 10000  # Minimum wealth to start a company
    POSSIBLE_MARKETS = list(range(0,10))
    # [
    # "Technology", "Healthcare", "Finance", "Education", "E-commerce",
    # "Energy", "Food & Beverage", "Manufacturing", "Media", "Entertainment",
    # "Art & Design"
    # ]

    # Math
    EPSILON = 1e-6


class Product(NamedObject):
    def __init__(self, company: 'Company', price: Union[int, float]=1, quality: Union[int, float]=1):
        self.quality = quality
        self.price = price
        self.company = company
    def __repr__(self):
        return f"{__class__.__name__}({self.__dict__})"


class ProductGroup(tuple):
    def __new__(cls, products):
        # Check if all elements is a valid product
        if not all(isinstance(p, Product) for p in products):
            raise ValueError("All elements in the product group must be of type Product.")

        # Create a new instance of the tuple
        return super().__new__(cls, products)
        
    def __init__(self, products: Tuple[Product]):
        # self.quality_min = min(self.all_quality)
        self.quality_max = max(self.all_quality)
        # self.quality_range = self.quality_max - self.quality_min
    
    @property
    def all_quality(self) -> tuple:
        return tuple(p.quality for p in self)
    @property
    def all_price(self) -> tuple:
        return tuple(p.quality for p in self)

    def get_quality_normalized(self, product: Product) -> float:
        return product.quality / self.quality_max


class NicheMarket:
    def __init__(self, field, demand, competition, profit_margin):
        self.field = field
        self.demand = demand
        self.competition = competition
        self.profit_margin = profit_margin

    @staticmethod
    def generate_niche_markets(num_markets):
        niche_markets = []
        for _ in range(num_markets):
            name = random.choice(Config.POSSIBLE_MARKETS)  # random market field
            demand = random.randint(50, 200)  # random demand
            competition = random.randint(10, 100)  # random competition
            profit_margin = round(random.uniform(0.1, 0.6), 2)  # random profit_margin
            niche_markets.append(NicheMarket(name, demand, competition, profit_margin))
        return niche_markets

    def calculate_attractiveness(self):
        return self.demand * self.profit_margin / (self.competition + 1)  # Avoid division by zero


class Individual(NamedObject, FundsObject, funds_precision=Config.FUNDS_PRECISION):
    def __init__(self, talent: float, initial_funds: float, skills: list, risk_tolerance: float ):
        self.set_funds(initial_funds)
        self.talent = talent
        self.employer: Optional[Company] = None
        self.salary = Config.FUNDS_PRECISION(0)
        self.owning_company: list[Company] = []
        self.risk_tolerance = risk_tolerance
        self.skills = skills

        self.expenses = 0

    @property
    def income(self):
        return self.salary + sum(c.dividend for c in self.owning_company)

    def make_purchase(self, product: Product):
        self.transfer_funds_to(product.company, product.price)
        product.company.revenue += product.price

    def score_product(self, product: Product) -> float:
        return np.tanh((self.funds + self.income) / product.price) * product.quality if self.can_afford(product.price) else 0

    def decide_purchase(self, products: ProductGroup) -> Optional[Product]:
        if not products:
            return None

    def choose_niche(self, niches):
        niches = NicheMarket.generate_niche_markets(500)
        best_niche = None
        best_score = 0
        for niche in niches:
            if niche.field in self.skills:
                score = niche.calculate_attractiveness() * self.risk_tolerance
                if score > best_score:
                    best_score = score
                    best_niche = niche
        return best_niche

        product_scores = np.array([self.score_product(p) for p in products])
        sum_product_scores = product_scores.sum()

        if sum_product_scores == 0:
            return None

        chosen_product: Product = np.random.choice(products, p=product_scores/sum_product_scores)
        return chosen_product

    def find_job(self, companies: List['Company']):
        if self.employer is None:
            for company in companies:
                if company.hire_employee(self, self.talent * Config.SALARY_FACTOR):
                    break
    
    def estimate_runout(self) -> Optional[int]:
        # Estimate how many time steps until the individual runs out of funds
        est = self.funds / (self.income - self.expenses)
        return int(est) if est > 0 else None

    def self_evaluate(self, products: List[Product]) -> float:
        # Evaluation based on how much funds the individual has and the product that they are purchasing
        pass


class Company(NamedObject, FundsObject, funds_precision=Config.FUNDS_PRECISION):
    def __init__(self, owner: Individual, initial_funds: float=0):
        self.set_funds(initial_funds)
        self.owner = owner
        owner.owning_company.append(self)
        
        self.employees: List[Individual] = []
        self.product = Product(self)
        self.revenue = Config.FUNDS_PRECISION(0)
        self.suppliers: List[Company] = []
        self.max_employees = random.randint(Config.MIN_COMPANY_SIZE, Config.MAX_COMPANY_SIZE)
        self.bankruptcy = False
        self.profit_margin = 0.2    # (1 + profit margin) * price * sales = revenue  TODO: smart margin

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

    def estimate_sales(self, population: int, company_count: int) -> float:
        # TODO: Smarter estimate sales 
        return population / company_count

    def update_product_attributes(self, population: int, company_count: int):
        # Update product quality and price
        self.product.quality = max(1, self.calculate_product_quality())  # Ensure quality >= 1
        self.product.price = max(1, self.costs * (1 + self.profit_margin)) / self.estimate_sales(population=population, company_count=company_count)  # Ensure price >= 1

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
            self.owner.owning_company.remove(self)
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
            f"Costs: {self.costs:.2f}, Revenue: {self.revenue:.2f}, Dividends: {self.dividend:.2f}, "
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
        risk_tolerance = [round(uniform(0.5, 2.0), 2) for _ in range(num_individuals)]
        skills = list[set(choices(Config.POSSIBLE_MARKETS, k=Config.MAX_SKILLS))]

        return [Individual(t, f, skills= s, risk_tolerance= r) for t, f, s, r in zip(talents, initial_funds, skills, risk_tolerance)]



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

    def get_all_products(self) -> ProductGroup:
        return ProductGroup([company.product for company in self.companies])

    def simulate_step(self):
        self.stats.step += 1
        # Update company product prices and quality and reset revenue
        for company in self.companies:
            company.update_product_attributes(population=len(self.individuals), company_count=len(self.companies))
            company.revenue = 0

        all_products = self.get_all_products()

        # Individuals spend money
        for individual in self.individuals:
            chosen_product = individual.decide_purchase(all_products)
            if chosen_product:
                individual.make_purchase(chosen_product)
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
                individual.find_job(self.companies)

        # Start new companies
        for individual in self.individuals:
            be_entrepreneur = Individual.choose_niche(individual, niches=Config.POSSIBLE_MARKETS)
            # TODO: Instead of random chance of starting a new company, consider the current market demands
            if individual.funds > Config.MIN_WEALTH_FOR_STARTUP and be_entrepreneur:
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
        num_individuals= 10000,
        num_companies=50,
        num_steps=1000,
        state_pickle_path="economy_simulation.pkl",
        resume_state=False,
    )
#10000


# if __name__ == "__main__":
#     markets = NicheMarket.generate_niche_markets(10)
#     num_individuals = 3
#     for market in markets:
#         print(f"Market: {market.field}, Demand: {market.demand}, "
#               f"Competition: {market.competition}, "
#               f"Profit Margin: {market.profit_margin}")













    # Load simulation state (optional)
    # economy = Economy.load_state("economy_simulation.pkl")

    # Plot results
    # plot_results(economy, save_path="economy_simulation_results.png")
    # print_summary(economy)