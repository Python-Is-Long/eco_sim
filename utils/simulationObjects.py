import random
from dataclasses import dataclass, fields
from typing import List, Optional, Union, Iterable, Callable, Set, Any
from typing import TYPE_CHECKING

import numpy as np

from utils.data import to_db_types, convert_value
from utils.genericObjects import NamedObject, FundsObject

if TYPE_CHECKING:
    from sim import Economy

@dataclass
class Reports:
    """A report dataclass to store data into a database.
    This dataclass will force all attributes to be converted to the correct type.

    Attributes:
        step (int): The current step of the simulation.
        name (str): The name of the object.
    """
    step: int
    name: str

    @staticmethod
    def table_name() -> str:
        """Returns what table should this report be stored in the database."""
        # Override this method in a subclass to specify the table name for that report
        pass

    @staticmethod
    def db_type_overrides() -> dict[str, str]:
        """Returns the type overrides for attributes to store the report."""
        # Override this method in a subclass to force a column in the database to be a specific type
        return {'step': 'UInt32', 'name': 'UUID'}

    def __post_init__(self):
        """Force conversion of all attributes to the correct type."""
        for field in fields(self): # type: ignore
            current_value = getattr(self, field.name)
            try:
                converted_value = convert_value(current_value, field.type)
            except Exception as e:
                raise TypeError(
                    f"Failed to convert field '{field.name}' from {type(current_value)} to {field.type}: {e}"
                ) from e
            setattr(self, field.name, converted_value)

    @classmethod
    def get_db_types(cls) -> dict[str, str]:
        """Returns the database types for each attribute in the report."""
        annotations = {}
        for c in cls.mro()[::-1]:
            try:
                annotations.update(c.__annotations__)
            except AttributeError:
                # object, at least, has no __annotations__ attribute.
                pass
        db_types = {attr: to_db_types(anno) for attr, anno in annotations.items()}
        db_types.update(cls.db_type_overrides())
        return db_types


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

    # Company settings
    INITIAL_FUNDS_COMPANY: Union[int, float] = 100000  # Increased initial funds
    MIN_COMPANY_SIZE: int = 5  # Minimum initial company size
    MAX_COMPANY_SIZE: int = 20  # Maximum initial company size
    SALARY_FACTOR: Union[int, float] = 100  # Salary = talent * SALARY_FACTOR
    DIVIDEND_RATE: float = 0.1  # 10% of profit paid as dividends
    PROFIT_MARGIN_FOR_HIRING: Union[int, float] = 1.5  # Higher margin for hiring

    # Entrepreneurship settings
    MAX_SKILLS = 3
    STARTUP_COST_FACTOR: float = 0.5  # Fraction of wealth used to start a company
    MIN_WEALTH_FOR_STARTUP: Union[int, float] = 100000  # Minimum wealth to start a company
    POSSIBLE_MARKETS = list(range(0, 10))

    # Math
    EPSILON = 1e-6

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class NicheMarket:
    def __init__(self, field, demand, competition, profit_margin):
        self.field = field
        self.demand = demand
        self.competition = competition
        self.profit_margin = profit_margin

    @staticmethod
    def generate_niche_markets(num_markets, possible_markets):
        niche_markets = []
        for _ in range(num_markets):
            name = random.choice(possible_markets)  # random market field
            demand = random.randint(50, 200)  # random demand
            competition = random.randint(10, 100)  # random competition
            profit_margin = round(random.uniform(0.1, 0.6), 2)  # random profit_margin
            niche_markets.append(NicheMarket(name, demand, competition, profit_margin))
        return niche_markets

    def calculate_attractiveness(self):
        return self.demand * self.profit_margin / (self.competition + 1)  # Avoid division by zero


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


@dataclass
class IndividualReports(Reports):
    # Make sure each attribute has valid type annotation
    funds: float
    income: float
    expenses: float
    salary: float
    talent: float
    risk_tolerance: float
    owning_company: list[str]

    @staticmethod
    def table_name():
        return 'individual'


class Individual(FundsObject, NamedObject):
    def __init__(
        self,
        eco: 'Economy',
        talent: float,
        initial_funds: float,
        skills: Set[int],
        risk_tolerance: float,
        configuration: Config = Config()
    ):
        self.config = configuration
        super().__init__(
            starting_funds=initial_funds,
            funds_precision=configuration.FUNDS_PRECISION
        )
        self.eco = eco

        self.talent = talent
        self.unemployed_state = 0
        self.employer: Optional[Company] = None
        self.salary = configuration.FUNDS_PRECISION(0)
        self.owning_company: list[Company] = []
        self.risk_tolerance = risk_tolerance
        self.skills = skills

        self.expenses = 0

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

    def choose_niche(self, niches):
        niches = NicheMarket.generate_niche_markets(500, self.config.POSSIBLE_MARKETS)
        best_niche = None
        best_score = 0
        for niche in niches:
            if niche.field in self.skills:
                score = niche.calculate_attractiveness() * self.risk_tolerance
                if score > best_score:
                    best_score = score
                    best_niche = niche
        return best_niche

    def decide_purchase(self, products: ProductGroup) -> Optional[Product]:
        if not products:
            return None

        product_scores = np.array([self.score_product(p) for p in products])
        sum_product_scores = product_scores.sum()

        if sum_product_scores == 0:
            return None

        chosen_product: Product = np.random.choice(products, p=product_scores / sum_product_scores)
        return chosen_product

    def find_opportunities(self):
        # Find jobs if the individual is not employed
        if self.employer is None:
            # ego_value = [1, 0.9, 0.8, 0.7, 0.6, 0.5] # change to a function instead of a list
            ego_value = [1, 0.95, 0.9, 0.85, 0.8, 0.75]
            # TODO: make the lowest ego value scale to the minimum product price (ppl will calculate minimum viable salary to survive)
            for company in self.eco.companies:
                if company.hire_employee(self, self.talent * self.config.SALARY_FACTOR * ego_value[self.unemployed_state]):
                    return
            if self.unemployed_state < len(ego_value) - 1:
                self.unemployed_state += 1

        # Start new company
        be_entrepreneur = self.choose_niche(niches=self.config.POSSIBLE_MARKETS)
        # TODO: Instead of random chance of starting a new company, consider the current market demands
        if self.funds > self.config.MIN_WEALTH_FOR_STARTUP and random.random() < self.eco.market_potential and be_entrepreneur:
            initial_funds = self.funds * self.config.STARTUP_COST_FACTOR
            # Add this new company to a queue to be created by the main thread
            self.eco.creating_companies.append(CompanyCreation(owner=self, initial_funds=initial_funds))

    def estimate_runout(self) -> Optional[int]:
        # Estimate how many time steps until the individual runs out of funds
        est = self.funds / (self.income - self.expenses)
        return int(est) if est > 0 else None

    def self_evaluate(self, products: List[Product]) -> float:
        # Evaluation based on how much funds the individual has and the product that they are purchasing
        pass

    def purchase_product(self):
        target_product = self.decide_purchase(self.eco.get_all_products())
        if target_product is not None:
            self.make_purchase(self.eco.companies, target_product)
            self.expenses = target_product.price

    def report(self):
        return IndividualReports(
            step=self.eco.stats.step,
            name=self.name,
            funds=self.funds,
            income=self.income,
            expenses=self.expenses,
            salary=self.salary,
            talent=self.talent,
            risk_tolerance=self.risk_tolerance,
            owning_company=[c.name for c in self.owning_company],
        )


@dataclass
class CompanyCreation:
    owner: Individual
    initial_funds: Any

@dataclass
class CompanyReports(Reports):
    # Make sure each attribute has valid type annotation
    owner: str
    funds: np.float64
    employees: list[str]
    product: str
    costs: float
    revenue: float
    profit: float
    dividend: float
    bankruptcy: bool

    @staticmethod
    def table_name():
        return 'company'


class Company(FundsObject, NamedObject):
    def __init__(
            self,
            eco: 'Economy',
            owner: Individual,
            initial_funds: float = 0,
            configuration: Config = Config()
    ):
        self.config = configuration
        super().__init__(
            starting_funds=initial_funds,
            funds_precision=self.config.FUNDS_PRECISION
        )
        self.eco = eco

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
        base_quality_employee = employee_contribution / (diminishing_factor_employee + self.config.EPSILON)

        # Additional quality from raw_materials
        base_quality_raw_material = 0
        if len(self.raw_materials) > 1:
            base_quality_raw_material = np.median(product.quality for product in self.raw_materials)
        return base_quality_employee + base_quality_raw_material

    @staticmethod
    def estimate_sales(population: int, company_count: int) -> float:
        # TODO: Smarter estimate sales
        return population / company_count

    def update_product_attributes(self, population: int, company_count: int):
        # Update product quality and price
        self.product.quality = max(1, self.calculate_product_quality())  # type: ignore
        self.product.price = (max(1, self.costs * (1 + self.profit_margin)) /  # type: ignore
                              self.estimate_sales(population=population, company_count=company_count))

    def hire_employee(self, candidate: Individual, salary: float) -> bool:
        #TODO: allow company to grow indefinitely
        if self.funds >= salary and len(self.employees) < self.max_employees:
            candidate.employer = self
            candidate.salary = salary
            self.employees.append(candidate)
            candidate.unemployed_state=0
            return True
        return False

    def fire_employee(self, employee: Individual):
        if employee in self.employees:
            self.employees.remove(employee)
            employee.employer = None
            employee.salary = 0.0

    # fix zero raw_material issue
    def find_raw_material(self, products: Iterable[Product]):
        # Get all products that does not have this company's product as a child material
        potential_materials = [p for p in products if self.product not in p.get_materials_recursive() and p is not self.product]

        if not potential_materials:
            return

        if self.product.quality == 1 or not self.raw_materials:
            self.raw_materials.append(random.choice(potential_materials))
        if len(self.raw_materials) > 0 and random.random() < np.log(1 / len(self.raw_materials) + self.config.EPSILON) * 10:
            self.raw_materials.append(random.choice(potential_materials))

    # fix forever growing raw_material issue
    def remove_raw_material(self):
        if len(self.raw_materials) > 1 and random.random() < 1 / np.log10(self.product.quality + self.config.EPSILON) / 10:
            self.raw_materials.remove(random.choice(self.raw_materials))

    def check_bankruptcy(self) -> bool:
        # TODO: if a company has no worker for over x step, bankrupt
        if self.funds < self.costs and len(self.raw_materials) > 0:
            # Reduce product cost before firing employees
            self.raw_materials.remove(sorted(list(self.raw_materials), key=lambda x: x.price, reverse=True)[0])

        return self.funds < self.costs

    def declare_bankruptcy(self):
        # Fire all employees
        for employee in self.employees:
            self.fire_employee(employee)
        # The owner runs with leftover company funds
        self.transfer_funds_to(self.owner, self.funds)
        self.bankruptcy = True
        self.eco.stats.num_bankruptcies += 1

    def update_product(self):
        # Update company product prices and quality and reset revenue
        self.remove_raw_material()  # see if remove raw_material at this step
        self.update_product_attributes(population=len(self.eco.individuals), company_count=len(self.eco.companies))
        self.revenue = 0

        # see if able to find a raw_material at this step
        self.find_raw_material(self.eco.get_all_products())

    def do_finance(self):
        # Pay dividends from profit to owner
        self.transfer_funds_to(self.owner, self.dividend)
        # Check for bankruptcy
        if self.check_bankruptcy():
            self.declare_bankruptcy()
            # Add this company to a queue to be removed by the main thread
            self.eco.removing_companies.append(self)

        # Pays employees salary
        [e.transfer_funds_from(self, e.salary) for e in self.employees]

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
            # TODO: add a bankruptcy index every time when there's no worker to fire

    def report(self):
        return CompanyReports(
            step=self.eco.stats.step,
            name=self.name,
            owner=self.owner.name,
            funds=self.funds,
            employees=[emp.name for emp in self.employees],
            product=self.product.name,
            costs=self.costs,
            revenue=self.revenue,
            profit=self.profit,
            dividend=self.dividend,
            bankruptcy=self.bankruptcy
        )

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