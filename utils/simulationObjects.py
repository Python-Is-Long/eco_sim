import random
from typing import List, Optional, Union, Iterable, Callable
from typing import TYPE_CHECKING

import mesa
import numpy as np
from mesa.agent import Agent

from utils.genericObjects import FundsObject

if TYPE_CHECKING:
    from sim import Economy


class Config:
    """Default Settings"""
    # Simulation settings
    NUM_INDIVIDUAL: int = 100
    NUM_COMPANY: int = 5
    SEED: int = 42
    FUNDS_PRECISION: Callable = np.float64  # Object used to store funds (int or float like)

    # Individual settings
    INITIAL_FUNDS_INDIVIDUAL: Union[int, float] = 5000  # Initial funds for individuals
    TALENT_MEAN: Union[int, float] = 100  # Mean talent (IQ-like)
    TALENT_STD: Union[int, float] = 15  # Standard deviation of talent

    # Company settings
    INITIAL_FUNDS_COMPANY: Union[int, float] = 100000  # Increased initial funds
    MIN_COMPANY_SIZE: int = 5  # Minimum initial company size
    MAX_COMPANY_SIZE: int = 20  # Maximum initial company size
    SALARY_FACTOR: Union[int, float] = 100  # Salary = talent * SALARY_FACTOR
    DIVIDEND_RATE: float = 0.1  # 10% of profit paid as dividends
    PROFIT_MARGIN_FOR_HIRING: Union[int, float] = 1.5  # Higher margin for hiring

    # Entrepreneurship settings
    STARTUP_COST_FACTOR: float = 0.5  # Fraction of wealth used to start a company
    MIN_WEALTH_FOR_STARTUP: Union[int, float] = 10000  # Minimum wealth to start a company

    # Math
    EPSILON = 1e-6

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class Product:
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
        self.company = company
        self.materials = ProductGroup(materials if materials else [])

    def __repr__(self):
        return f"{__class__.__name__}(price={self.price}, quality={self.quality})"

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


class Individual(FundsObject, Agent):
    model: 'Economy'

    def __init__(self, model: 'mesa.model', talent: float, initial_funds: float, configuration: Config = Config):
        self.config = configuration
        super().__init__(
            model=model,
            starting_funds=initial_funds,
            funds_precision=configuration.FUNDS_PRECISION
        )
        self.talent = talent
        self.employer: Optional[Company] = None
        self.salary = configuration.FUNDS_PRECISION(0)
        self.owning_company: list[Company] = []

        self.expenses = 0

    @property
    def income(self):
        return self.salary + sum(c.dividend for c in self.owning_company)

    def estimate_runout(self) -> Optional[int]:
        # Estimate how many time steps until the individual runs out of funds
        est = self.funds / (self.income - self.expenses)
        return int(est) if est > 0 else None

    def self_evaluate(self, products: List[Product]) -> float:
        # Evaluation based on how much funds the individual has and the product that they are purchasing
        pass

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

    def purchase_product(self, products: ProductGroup):
        target_product = self.decide_purchase(products)
        if target_product is not None:
            self.transfer_funds_to(target_product.company, target_product.price)
            target_product.company.revenue += target_product.price
            self.expenses = target_product.price

    def find_job(self, companies: List['Company']):
        if self.employer is None:
            for company in companies:
                if company.hire_employee(self, self.talent * self.config.SALARY_FACTOR):
                    break

    def start_new_company(self):
        # The chance of starting a new company decreases with as the current number of companies increases
        market_potential = np.log10(len(self.model.agents_by_type[Individual])) / len(self.model.agents_by_type[Company])
        if random.random() < market_potential:
            initial_funds = self.funds * self.config.STARTUP_COST_FACTOR
            new_company = Company(model=self.model, owner=self)
            self.transfer_funds_to(new_company, initial_funds)
            self.model.num_new_companies += 1  # Increment new company counter


class Company(FundsObject, Agent):
    model: 'Economy'

    def __init__(
            self,
            model: 'mesa.model',
            owner: Individual,
            initial_funds: float = 0,
            configuration: Config = Config,
    ):
        self.config = configuration
        super().__init__(
            model=model,
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
            base_quality_raw_material = np.median([product.quality for product in self.raw_materials])
        return base_quality_employee + base_quality_raw_material

    def estimate_sales(self, population: int, company_count: int) -> float:
        # TODO: Smarter estimate sales
        return population / company_count

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

    # fix zero raw_material issue
    def find_raw_material(self):
        all_products = self.model.get_all_products()
        # Filter all the products that made of part of this company's own product or this company's own product
        potential_materials = [p for p in all_products if
                               self.product not in p.get_materials_recursive() and p != self.product]

        # TODO: smarter material selection
        if self.product.quality == 1 or not self.raw_materials:
            self.raw_materials.append(random.choice(potential_materials))
        if len(self.raw_materials) > 0 and random.random() < np.log(1 / len(self.raw_materials) + self.config.EPSILON):
            self.raw_materials.append(random.choice(potential_materials))

    # fix forever growing raw_material issue
    def remove_raw_material(self):
        if len(self.raw_materials) > 1 and random.random() < 1 / np.log10(self.product.quality + self.config.EPSILON):
            self.raw_materials.remove(random.choice(self.raw_materials))

    def update_product_attributes(self):
        population = len(self.model.agents_by_type[Individual])
        company_count = len(self.model.agents_by_type[Company])
        # Update product quality and price
        self.product.quality = max(1, self.calculate_product_quality())  # Ensure quality >= 1
        self.product.price = (max(1, self.costs * (1 + self.profit_margin)) /
                              self.estimate_sales(population=population,
                                                  company_count=company_count))  # Ensure price >= 1

    def adjust_workforce(self, unemployed: list[Individual]):
        if self.revenue > self.costs * self.config.PROFIT_MARGIN_FOR_HIRING:
            # Hire new employees
            if unemployed:
                new_employee = max(unemployed, key=lambda x: x.talent)
                self.hire_employee(new_employee, new_employee.talent * self.config.SALARY_FACTOR)
        elif self.revenue < self.costs:
            # Fire employees
            if self.employees:
                employee_to_fire = random.choice(self.employees)
                self.fire_employee(employee_to_fire)

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

    def do_finances(self):
        # Pay dividends from profit to owner
        self.transfer_funds_to(self.owner, self.dividend)
        # Check for bankruptcy
        if self.check_bankruptcy():
            self.model.num_bankruptcies += 1
            self.remove()
        # Pays employees salary
        [e.transfer_funds_from(self, e.salary) for e in self.employees]

    def print_statistics(self):
        stats = (
            f"Agent ID: {self.unique_id}, "
            f"Owner ID: {self.owner.unique_id}, Funds: {self.funds:.2f}, "
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
