import random
import heapq
from typing import List, Optional, Union, Iterable, Callable

import numpy as np
from packaging.utils import canonicalize_name

from utils.genericObjects import NamedObject, FundsObject
from utils.calculation import calculate_choice_probabilities


class Config:
    """Default Settings"""
    # Simulation settings
    NUM_INDIVIDUAL: int = 100
    NUM_COMPANY: int = 5
    FUNDS_PRECISION: Callable = np.float64  # Object used to store funds (int or float like)

    # Individual settings
    INITIAL_FUNDS_INDIVIDUAL: Union[int, float] = 5000  # Initial funds for individuals
    TALENT_MEAN: Union[int, float] = 100  # Mean talent (IQ-like)
    TALENT_STD: Union[int, float] = 15  # Standard deviation of talent
    TEMPERATURE_JOB_OFFER: float = 7.5 # lower temperature = more likely goes to higher pay job

    # Company settings
    INITIAL_FUNDS_COMPANY: Union[int, float] = 100000  # Increased initial funds
    MIN_COMPANY_SIZE: int = 5  # Minimum initial company size
    MAX_COMPANY_SIZE: int = 20  # Maximum initial company size
    SALARY_FACTOR: Union[int, float] = 100  # Salary = talent * SALARY_FACTOR
    DIVIDEND_RATE: float = 0.1  # 10% of profit paid as dividends
    PROFIT_MARGIN_FOR_HIRING: Union[int, float] = 1.5  # Higher margin for hiring
    MIN_SALARY_THRESHOLD = 4500
    MAX_SALARY_THRESHOLD = 15000

    # Entrepreneurship settings
    STARTUP_COST_FACTOR: float = 0.5  # Fraction of wealth used to start a company
    MIN_WEALTH_FOR_STARTUP: Union[int, float] = 10000  # Minimum wealth to start a company

    # Math
    EPSILON = 1e-6

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class Product(NamedObject):
    def __init__(
        self,
        company: 'Company',
        price: int | float = 1,
        quality: int | float = 1,
        materials: Iterable['Product'] = None
    ):
        super().__init__()

        self.quality = quality
        self.price = price
        self.company = company.name
        self.materials = ProductGroup(materials if materials else [])

    def __repr__(self):
        return f"{__class__.__name__}(name={self.name}, price={self.price}, quality={self.quality})"

    @property
    def cost(self) -> float:
        return self.materials.total_price

    def get_materials_recursive(self, known_child=None):
        if known_child is None:
            known_child = set()

        for material in self.materials:
            # When a material is not already known, add it to the set and get its materials
            if material not in known_child:
                known_child.add(material)
                material.get_materials_recursive(known_child)
        return known_child


class ProductGroup(tuple):
    """A tuple of products with additional methods for managing products."""

    def __new__(cls, products: Iterable[Product]):
        # Check if all elements is a valid product
        if not all(isinstance(p, Product) for p in products):
            raise ValueError("All elements in the product group must be of type Product.")

        # Create a new instance of the tuple
        return super().__new__(cls, products)

    def __init__(self, products: Iterable[Product]):
        if self:
            self.quality_max = max(self.all_quality)

    @property
    def all_quality(self) -> tuple:
        return tuple(p.quality for p in self)

    @property
    def all_price(self) -> tuple:
        return tuple(p.quality for p in self)

    @property
    def total_price(self) -> float:
        return sum(self.all_price)

    def get_quality_normalized(self, product: Product) -> float:
        if not self:
            raise ValueError("Product group is empty.")

        return product.quality / self.quality_max


class Individual(FundsObject, NamedObject):
    def __init__(self, talent: float, initial_funds: float, configuration: Config = Config):
        self.config = configuration
        super().__init__(
            starting_funds=initial_funds,
            funds_precision=configuration.FUNDS_PRECISION
        )
        self.talent = talent
        self.employer: Optional[Company] = None
        self.salary = configuration.FUNDS_PRECISION(0)
        self.owning_company: list[Company] = []

        self.expenses = 0

    def evaluate_opportunity(self, base_probability: float) -> bool:
        """Determine if individuals can be entrepreneurs"""
        wealth_factor = min(1.0, self.funds / (self.config.MIN_WEALTH_FOR_STARTUP * 2))
        talent_factor = np.tanh(self.talent / 150)

        startup_probability = base_probability * (0.5 + 0.3 * wealth_factor + 0.2 * talent_factor)
        # 0.5 base chance = 50% # 0.3 wealth influence -> 30% # 0.2 -> talent influence ->20%
        return np.random.random() < startup_probability

    def start_company(self, all_individuals: List['Individual']) -> Optional['Company']:
        """Individual decides to start a business and establish a company"""
        if self.funds < self.config.MIN_WEALTH_FOR_STARTUP:
            return None

        initial_funds = self.funds * self.config.STARTUP_COST_FACTOR
        new_company = Company(self, initial_funds)
        self.transfer_funds_to(new_company, initial_funds)

        # recruiting
        available_workers = [ind for ind in all_individuals if ind != self and ind.employer is None]
        num_initial_employees = min(new_company.max_employees, len(available_workers), random.randint(1, 10))
        potential_employees = sorted(available_workers, key=lambda x: x.talent, reverse=True)[:num_initial_employees]

        for employee in potential_employees:
            initial_salary = 50 + employee.talent * 0.5
            if new_company.hire_employee(employee, initial_salary):
                available_workers.remove(employee)

        return new_company

    @property
    def income(self):
        return self.salary + sum(c.dividend for c in self.owning_company)

    def make_purchase(self, companies: List['Company'], product: Product):
        for c in companies:
            if product.company is c.name:
                self.transfer_funds_to(c, product.price)
                c.revenue += product.price

    def score_product(self, product: Product) -> float:
        return np.tanh((self.funds + self.income) / product.price) * product.quality if self.can_afford(
            product.price) else 0

    def decide_purchase(self, products: ProductGroup) -> Optional[Product]:
        if not products:
            return None

        product_scores = np.array([self.score_product(p) for p in products])
        sum_product_scores = product_scores.sum()

        if sum_product_scores == 0:
            return None

        chosen_product: Product = np.random.choice(products, p=product_scores / sum_product_scores)
        return chosen_product

    def find_job(self, companies: List['Company']):
        if self.employer:
            current_salary = self.salary
            better_offers = []
            for company in companies:
                company_offered = company.calculate_salary_offer(self)
                if company_offered and company_offered > current_salary:
                    better_offers.append((company_offered, company))

            if better_offers:
                better_offers.sort(reverse=True, key=lambda x:x[0])
                best_offer_salary, best_company = better_offers[0]

                if random.random() < 0.5 or (best_offer_salary > current_salary * 1.2):
                    self.employer.fire_employee(self)
                    best_company.hire_employee(self, salary=best_offer_salary)
            return

        companies_offering: List[Company] = []
        job_offers: List[float] = []
        # saving all the offers
        for company in companies:
            if offer := company.calculate_salary_offer(self):
                companies_offering.append(company)
                job_offers.append(offer)
        if job_offers:
            possibilities = calculate_choice_probabilities(offers=job_offers, temperature=Config.TEMPERATURE_JOB_OFFER)
            chosen_index = np.random.choice(range(len(possibilities)), p=possibilities)
            companies_offering[chosen_index].hire_employee(self, salary=job_offers[chosen_index])

        # if self.employer is None:
        #     for company in companies:
        #         if company.hire_employee(self, self.talent * self.config.SALARY_FACTOR):
        #             break

    def estimate_runout(self) -> Optional[int]:
        # Estimate how many time steps until the individual runs out of funds
        est = self.funds / (self.income - self.expenses)
        return int(est) if est > 0 else None

    def self_evaluate(self, products: List[Product]) -> float:
        # Evaluation based on how much funds the individual has and the product that they are purchasing
        pass


class Company(FundsObject, NamedObject):
    def __init__(self, owner: Individual, initial_funds: float = 0, configuration: Config = Config):
        self.config = configuration
        super().__init__(
            starting_funds=initial_funds,
            funds_precision=self.config.FUNDS_PRECISION
        )

        self.set_funds(initial_funds)
        self.owner = owner
        owner.owning_company.append(self)

        self.employees: List[Individual] = []
        self.product = Product(self)
        self.revenue = self.config.FUNDS_PRECISION(0)
        self.raw_materials: List[Product] = []
        self.max_employees = random.randint(self.config.MIN_COMPANY_SIZE, self.config.MAX_COMPANY_SIZE)
        self.bankruptcy = False
        self.profit_margin = 0.2  # (1 + profit margin) * price * sales = revenue  TODO: smart margin
        self.min_salary_threshold = Config.MIN_SALARY_THRESHOLD
        self.max_salary_threshold = Config.MAX_SALARY_THRESHOLD


    @property
    def total_salary(self):
        return sum(emp.salary for emp in self.employees)

    @property
    def total_material_cost(self):
        return sum(product.price for product in self.raw_materials)

    @property
    def costs(self):
        return self.total_salary + self.total_material_cost

    @property
    def profit(self):
        return self.revenue - self.costs

    @property
    def dividend(self):
        return self.profit * self.config.DIVIDEND_RATE if self.profit > 0 else 0

    def calculate_product_quality(self) -> float:
        if not self.employees:
            return 0.0

        # Base quality from employees with diminishing returns
        employee_contribution = sum(emp.talent for emp in self.employees)
        diminishing_factor_employee = np.log(len(self.employees) + 1)
        base_quality_employee = employee_contribution / (diminishing_factor_employee + Config.EPSILON)

        # Additional quality from raw_materials
        base_quality_raw_material = 0
        if len(self.raw_materials) > 1:
            base_quality_raw_material = np.median(product.quality for product in self.raw_materials)
        return base_quality_employee + base_quality_raw_material

    def estimate_sales(self, population: int, company_count: int) -> float:
        # TODO: Smarter estimate sales
        return population / company_count

    def update_product_attributes(self, population: int, company_count: int):
        # Update product quality and price
        self.product.quality = max(1, self.calculate_product_quality())  # Ensure quality >= 1
        self.product.price = (max(1, self.costs * (1 + self.profit_margin)) /
                              self.estimate_sales(population=population, company_count=company_count))  # Ensure price >= 1

    def hire_employee(self, candidate: Individual, salary: float) -> bool:
        if len(self.employees) >= self.max_employees:
            return False # company full

        offered_salary = self.calculate_salary_offer(candidate=candidate)
        if offered_salary is None or self.funds < offered_salary:
            return False

        # 計算性價比：才能值 / 薪資
        talent_to_salary_ratio = candidate.talent / offered_salary if offered_salary > 0 else 0

        # 公司會計算目前雇用員工的平均性價比
        if self.employees:
            avg_ratio = (sum(emp.talent / emp.salary for emp in self.employees) / len(
                self.employees)) if self.employees else None
        else:
            avg_ratio = 0  # 沒有員工時，不比較性價比

        # 如果新員工的性價比比目前員工的平均值還低，則有 50% 機率不錄用
        if avg_ratio is not None and talent_to_salary_ratio < avg_ratio and random.random() > 0.5:
            return False  # 為了保留高性價比，50% 機率不錄用

        # 確保公司不會把所有錢都用來雇用 1-2 個超高薪的員工
        if len(self.employees) > 0 and offered_salary > self.funds * 0.3:
            return False  # 不能讓單一員工的薪資超過總資金的 30%

        # 選擇最佳員工（基於性價比）
        self.employees.append(candidate)
        candidate.employer = self
        candidate.salary = offered_salary
        self.funds -= offered_salary
        return True

        # if self.funds >= salary and len(self.employees) < self.max_employees:
        #     candidate.employer = self
        #     candidate.salary = salary
        #     self.employees.append(candidate)
        #     return True
        # return False

    def calculate_salary_offer(self, candidate: Individual, salary_factor: Union[int, float] = Config.SALARY_FACTOR) -> Optional[float]:
        base_salary = candidate.talent * np.random.uniform(80, salary_factor)
        # 公司會根據市場狀況（可用資金 & 已雇用人數）微調薪資
        demand_factor = 1.0 + (0.1 * (self.max_employees - len(self.employees)) / self.max_employees)  # 需求越大薪水越高
        financial_factor = min(1.2, max(0.8, self.funds / (self.max_employees * self.max_salary_threshold)))  # 根據資金調整
        negotiation_factor = np.random.uniform(0.9, 1.1)  # 隨機調整，模擬談判

        # 計算最終薪資
        expected_salary = base_salary * demand_factor * financial_factor * negotiation_factor

        # 如果薪資過低，調整為最低薪資門檻
        # if expected_salary < self.min_salary_threshold:
        expected_salary = self.min_salary_threshold

        # 如果薪資超過最大範圍，不考慮
        if expected_salary < self.min_salary_threshold:
            expected_salary = self.min_salary_threshold
        if expected_salary > self.max_salary_threshold or self.funds < expected_salary:
            return None

        return expected_salary

    def fire_employee(self, employee: Individual):
        if employee in self.employees:
            self.employees.remove(employee)
            employee.employer = None
            employee.salary = 0.0

    # fix zero raw_material issue
    def find_raw_material(self, products: List[Product]):
        # Get all products that does not have this company's product as a child material
        potential_materials = [p for p in products if self.product not in p.get_materials_recursive()]

        if self.product.quality == 1 or not self.raw_materials:
            self.raw_materials.append(random.choice(potential_materials))
        if len(self.raw_materials) > 0 and random.random() < np.log(1 / len(self.raw_materials) + Config.EPSILON) * 10:
            self.raw_materials.append(random.choice(potential_materials))

    # fix forever growing raw_material issue
    def remove_raw_material(self):
        if len(self.raw_materials) > 1 and random.random() < 1 / np.log10(self.product.quality + Config.EPSILON) / 10:
            self.raw_materials.remove(random.choice(self.raw_materials))

    def check_bankruptcy(self) -> bool:
        if self.funds < self.costs and len(self.raw_materials) > 0:
            # Reduce product cost before firing employees
            self.raw_materials.remove(sorted(list(self.raw_materials), key=lambda x: x.price, reverse=True)[0])

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
            f"Raw Materials: {len(self.raw_materials)}, "
            f"Product Quality: {self.product.quality:.2f}, Product Price: {self.product.price:.2f}, "
            f"Costs: {self.costs:.2f}, Revenue: {self.revenue:.2f}, Dividends: {self.dividend:.2f}, "
            f"Bankruptcy: {self.bankruptcy}"
        )
        print(stats)


if __name__ == "__main__":
    # Test products get child materials
    product1 = Product(None, 1, 1)
    product2 = Product(None, 2, 2)
    product1.materials = ProductGroup([product2])
    product2.materials = ProductGroup([product1])
    product3 = Product(None, 3, 3)
    product4 = Product(None, 4, 4, [product3])

    print(product1.get_materials_recursive())
    print(product4.get_materials_recursive())