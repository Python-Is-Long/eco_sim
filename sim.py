from dataclasses import dataclass
import numpy as np
from typing import List, Optional
import random
import matplotlib.pyplot as plt
from tqdm import tqdm


# Configuration
class Config:
    INITIAL_FUNDS_INDIVIDUAL = 1000  # Initial funds for individuals
    INITIAL_FUNDS_COMPANY = 10000  # Initial funds for companies
    SALARY_FACTOR = 100  # Salary = talent * SALARY_FACTOR
    DIVIDEND_RATE = 0.1  # 10% of profit paid as dividends
    SPENDING_PROBABILITY_FACTOR = 5000  # Wealth factor for spending probability
    STARTUP_COST_FACTOR = 0.5  # Fraction of wealth used to start a company
    MIN_WEALTH_FOR_STARTUP = 10000  # Minimum wealth to start a company
    PROFIT_MARGIN_FOR_HIRING = 1.2  # Revenue > costs * PROFIT_MARGIN_FOR_HIRING to hire
    BANKRUPTCY_THRESHOLD = 0  # Funds <= BANKRUPTCY_THRESHOLD means bankruptcy


@dataclass
class Product:
    quality: float
    price: float
    company: 'Company'


class Individual:
    def __init__(self, talent: float, initial_funds: float):
        self.talent = talent
        self.wallet = initial_funds
        self.employer: Optional[Company] = None
        self.salary = 0.0

    def can_afford(self, amount: float) -> bool:
        return self.wallet >= amount

    def spend(self, amount: float) -> bool:
        if self.can_afford(amount):
            self.wallet -= amount
            return True
        return False

    def receive_money(self, amount: float):
        self.wallet += amount

    def decide_purchase(self, products: List[Product]) -> Optional[Product]:
        if not products:
            return None

        # Wealthy individuals care more about quality than price
        wealth_factor = np.tanh(self.wallet / Config.SPENDING_PROBABILITY_FACTOR)

        def score_product(product: Product) -> float:
            quality_weight = 0.5 + 0.5 * wealth_factor
            price_weight = 1.5 - 0.5 * wealth_factor
            return quality_weight * product.quality - price_weight * product.price

        scored_products = [(score_product(p), p) for p in products]
        if not scored_products:
            return None

        best_product = max(scored_products, key=lambda x: x[0])[1]
        if self.can_afford(best_product.price):
            return best_product
        return None

    def find_job(self, companies: List['Company']):
        if self.employer is None:
            for company in companies:
                if company.hire_employee(self, self.talent * Config.SALARY_FACTOR):
                    break


class Company:
    def __init__(self, owner: Individual, initial_funds: float):
        self.owner = owner
        self.funds = initial_funds
        self.employees: List[Individual] = []
        self.product_quality = 0.0
        self.product_price = 0.0
        self.costs = 0.0
        self.revenue = 0.0
        self.suppliers: List[Company] = []
        self.max_employees = random.randint(5, 20)  # Random initial company size

    def calculate_product_quality(self) -> float:
        if not self.employees:
            return 0.0

        # Base quality from employees with diminishing returns
        employee_contribution = sum(emp.talent for emp in self.employees)
        diminishing_factor = np.log(len(self.employees) + 1)
        base_quality = employee_contribution / diminishing_factor

        # Additional quality from suppliers
        supplier_quality = sum(supplier.product_quality for supplier in self.suppliers)
        return base_quality + supplier_quality * 0.5

    def update_financials(self):
        total_salary = sum(emp.salary for emp in self.employees)
        supplier_costs = sum(supplier.product_price for supplier in self.suppliers)
        self.costs = total_salary + supplier_costs

        # Update product quality and price
        self.product_quality = self.calculate_product_quality()
        self.product_price = self.costs * 1.2  # 20% markup

    def hire_employee(self, candidate: Individual, salary: float) -> bool:
        if self.funds >= salary and len(self.employees) < self.max_employees:
            candidate.employer = self
            candidate.salary = salary
            self.employees.append(candidate)
            self.funds -= salary
            return True
        return False

    def fire_employee(self, employee: Individual):
        if employee in self.employees:
            self.employees.remove(employee)
            employee.employer = None
            employee.salary = 0.0

    def get_product(self) -> Product:
        return Product(
            quality=self.product_quality,
            price=self.product_price,
            company=self
        )

    def pay_dividends(self):
        profit = self.revenue - self.costs
        if profit > 0:
            dividend = profit * Config.DIVIDEND_RATE
            self.owner.receive_money(dividend)
            self.funds -= dividend

    def check_bankruptcy(self) -> bool:
        if self.funds <= Config.BANKRUPTCY_THRESHOLD:
            for employee in self.employees[:]:
                self.fire_employee(employee)
            return True
        return False


class EconomyStats:
    def __init__(self):
        self.wealth_gini = []
        self.avg_wealth = []
        self.num_companies = []
        self.unemployment_rate = []
        self.avg_product_quality = []
        self.avg_product_price = []
        self.num_bankruptcies = 0
        self.num_new_companies = 0

    def calculate_gini(self, wealths: List[float]) -> float:
        sorted_wealths = sorted(wealths)
        n = len(sorted_wealths)
        if n == 0:
            return 0
        index = np.arange(1, n + 1)
        return ((2 * index - n - 1) * sorted_wealths).sum() / (n * sum(sorted_wealths))


class Economy:
    def __init__(self, num_individuals: int, num_companies: int):
        self.individuals = self._create_individuals(num_individuals)
        self.companies = self._create_companies(num_companies)
        self.stats = EconomyStats()

    def _create_individuals(self, num_individuals: int) -> List[Individual]:
        talents = np.random.normal(100, 15, num_individuals)
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

    def simulate_step(self):
        # Update company financials and pay salaries
        for company in self.companies:
            company.update_financials()
            for employee in company.employees:
                if company.funds >= employee.salary:
                    company.funds -= employee.salary
                    employee.receive_money(employee.salary)
                else:
                    company.fire_employee(employee)

            # Pay dividends to owner
            company.pay_dividends()

            # Check for bankruptcy
            if company.check_bankruptcy():
                self.companies.remove(company)
                self.stats.num_bankruptcies += 1

        # Individuals spend money
        for individual in self.individuals:
            if random.random() < np.tanh(individual.wallet / Config.SPENDING_PROBABILITY_FACTOR):
                available_products = [company.get_product() for company in self.companies]
                chosen_product = individual.decide_purchase(available_products)
                if chosen_product:
                    if individual.spend(chosen_product.price):
                        chosen_product.company.revenue += chosen_product.price

        # Individuals find jobs
        for individual in self.individuals:
            if individual.employer is None:
                individual.find_job(self.companies)

        # Start new companies
        for individual in self.individuals:
            if individual.wallet > Config.MIN_WEALTH_FOR_STARTUP:
                self.start_new_company(individual)

        # Adjust workforce for companies
        for company in self.companies:
            self.adjust_workforce(company)

        # Collect statistics
        self.collect_statistics()

    def start_new_company(self, individual: Individual):
        initial_funds = individual.wallet * Config.STARTUP_COST_FACTOR
        individual.wallet -= initial_funds
        new_company = Company(individual, initial_funds)
        self.companies.append(new_company)
        self.stats.num_new_companies += 1

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

    def collect_statistics(self):
        wealths = [ind.wallet for ind in self.individuals]
        employed = len([ind for ind in self.individuals if ind.employer])

        self.stats.wealth_gini.append(self.stats.calculate_gini(wealths))
        self.stats.avg_wealth.append(np.mean(wealths))
        self.stats.num_companies.append(len(self.companies))
        self.stats.unemployment_rate.append(1 - employed / len(self.individuals))

        if self.companies:
            self.stats.avg_product_quality.append(np.mean([c.product_quality for c in self.companies]))
            self.stats.avg_product_price.append(np.mean([c.product_price for c in self.companies]))
        else:
            self.stats.avg_product_quality.append(0)
            self.stats.avg_product_price.append(0)


def run_simulation(num_individuals: int = 1000, num_companies: int = 50, num_steps: int = 1000) -> Economy:
    economy = Economy(num_individuals, num_companies)
    for _ in tqdm(range(num_steps), desc="Simulating economy"):
        economy.simulate_step()
    return economy


def plot_results(economy: Economy):
    fig, axes = plt.subplots(4, 2, figsize=(15, 16))
    fig.suptitle('Economic Simulation Results')

    axes[0, 0].plot(economy.stats.wealth_gini)
    axes[0, 0].set_title('Wealth Inequality (Gini)')
    axes[0, 0].set_ylabel('Gini Coefficient')

    axes[0, 1].plot(economy.stats.avg_wealth)
    axes[0, 1].set_title('Average Wealth')
    axes[0, 1].set_ylabel('Wealth')

    axes[1, 0].plot(economy.stats.num_companies)
    axes[1, 0].set_title('Number of Companies')
    axes[1, 0].set_ylabel('Count')

    axes[1, 1].plot(economy.stats.unemployment_rate)
    axes[1, 1].set_title('Unemployment Rate')
    axes[1, 1].set_ylabel('Rate')

    axes[2, 0].plot(economy.stats.avg_product_quality)
    axes[2, 0].set_title('Average Product Quality')
    axes[2, 0].set_ylabel('Quality')

    axes[2, 1].plot(economy.stats.avg_product_price)
    axes[2, 1].set_title('Average Product Price')
    axes[2, 1].set_ylabel('Price')

    axes[3, 0].plot(range(len(economy.stats.wealth_gini)), [economy.stats.num_bankruptcies] * len(economy.stats.wealth_gini), label='Bankruptcies')
    axes[3, 0].set_title('Number of Bankruptcies Over Time')
    axes[3, 0].set_ylabel('Count')

    axes[3, 1].plot(range(len(economy.stats.wealth_gini)), [economy.stats.num_new_companies] * len(economy.stats.wealth_gini), label='New Companies')
    axes[3, 1].set_title('Number of New Companies Over Time')
    axes[3, 1].set_ylabel('Count')

    plt.tight_layout()
    plt.show()


def print_summary(economy: Economy):
    print("\nFinal Economic Statistics:")
    print(f"Number of individuals: {len(economy.individuals)}")
    print(f"Number of companies: {len(economy.companies)}")
    print(f"Average wealth: {economy.stats.avg_wealth[-1]:.2f}")
    print(f"Wealth inequality (Gini): {economy.stats.wealth_gini[-1]:.2f}")
    print(f"Unemployment rate: {economy.stats.unemployment_rate[-1]:.2%}")
    print(f"Total bankruptcies: {economy.stats.num_bankruptcies}")
    print(f"Total new companies: {economy.stats.num_new_companies}")

    wealth_percentiles = np.percentile([ind.wallet for ind in economy.individuals], [25, 50, 75, 90, 99])
    print("\nWealth Distribution:")
    print(f"25th percentile: {wealth_percentiles[0]:.2f}")
    print(f"Median: {wealth_percentiles[1]:.2f}")
    print(f"75th percentile: {wealth_percentiles[2]:.2f}")
    print(f"90th percentile: {wealth_percentiles[3]:.2f}")
    print(f"99th percentile: {wealth_percentiles[4]:.2f}")


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    economy = run_simulation(num_individuals=10000, num_companies=50, num_steps=1000)
    plot_results(economy)
    print_summary(economy)