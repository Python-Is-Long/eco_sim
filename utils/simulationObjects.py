import random
from dataclasses import dataclass, fields
from typing import List, Optional, Union, Iterable, Callable, Set, Any
from typing import TYPE_CHECKING

import numpy as np


from utils.genericObjects import Agent, FundsObject
from utils.simulationUtils import Reports, AgentUpdates, SimulationAgents


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


class Product(Agent):
    def __init__(
        self,
        company: 'Company',
        price: int | float = 1,
        quality: int | float = 1,
        materials: Optional[Iterable['Product']] = None
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


class Individual(FundsObject):
    all_agents: SimulationAgents
    def __init__(
        self,
        all_agents: SimulationAgents,
        talent: float,
        initial_funds: float,
        skills: Set[int],
        risk_tolerance: float,
        configuration: Config = Config()
    ):
        self.config = configuration
        self.all_agents = all_agents
        self.all_agents.add(self)

        super().__init__(
            starting_funds=initial_funds,
            funds_precision=configuration.FUNDS_PRECISION
        )

        self.update = AgentUpdates(all_agents)
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

    def get_market_potential(self):
        # TODO: Handles case where len(self.companies) is 0
        return np.log10(len(self.all_agents[Individual])) / len(self.all_agents[Company])

    # Stage method
    def find_opportunities(self) -> bool:
        self.update.clear_history()
        # Find jobs if the individual is not employed
        if self.employer is None:
            # ego_value = [1, 0.9, 0.8, 0.7, 0.6, 0.5] # change to a function instead of a list
            ego_value = [1, 0.95, 0.9, 0.85, 0.8, 0.75]
            # TODO: make the lowest ego value scale to the minimum product price (ppl will calculate minimum viable salary to survive)
            for company in self.all_agents[Company]:
                if company.hire_employee(self, self.talent * self.config.SALARY_FACTOR * ego_value[self.unemployed_state]):
                    return True
            if self.unemployed_state < len(ego_value) - 1:
                self.update.attr_update(self, 'unemployed_state', self.unemployed_state + 1)

        # Start new company
        be_entrepreneur = self.choose_niche(niches=self.config.POSSIBLE_MARKETS)
        # TODO: Instead of random chance of starting a new company, consider the current market demands
        if self.funds > self.config.MIN_WEALTH_FOR_STARTUP and random.random() < self.get_market_potential() and be_entrepreneur:
            initial_funds = self.funds * self.config.STARTUP_COST_FACTOR
            # Add this new company to a queue to be created by the main thread
            self.update.add_agent(Company, owner=self, initial_funds=initial_funds)
        return True

    def estimate_runout(self) -> Optional[int]:
        # Estimate how many time steps until the individual runs out of funds
        est = self.funds / (self.income - self.expenses)
        return int(est) if est > 0 else None

    def self_evaluate(self, products: List[Product]) -> float:
        # Evaluation based on how much funds the individual has and the product that they are purchasing
        pass

    # Stage method
    def purchase_product(self) -> bool:
        self.update.clear_history()
        target_product = self.decide_purchase(ProductGroup(self.all_agents[Product]))
        if target_product is not None:
            self.make_purchase(self.all_agents[Company], target_product)
            self.update.attr_update(self, 'expenses', target_product.price)
        return True

    def report(self):
        return IndividualReports(
            step=self.step,
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


class Company(FundsObject):
    all_agents: SimulationAgents
    owner: Individual
    def __init__(
        self,
        all_agents: SimulationAgents,
        owner: Individual,
        initial_funds: float = 0,
        configuration: Config = Config()
    ):
        self.config = configuration
        self.all_agents = all_agents
        self.all_agents.add(self)
        super().__init__(
            starting_funds=initial_funds,
            funds_precision=self.config.FUNDS_PRECISION
        )
        self.update = AgentUpdates(all_agents)
        self.set_funds(initial_funds)
        self.owner = owner
        self.update.agent_list_update(owner, 'owning_company', 'append', self)

        self.employees: List[Individual] = []
        self.product = Product(self)
        self.all_agents.add(self.product)
        self.revenue = self.config.FUNDS_PRECISION(0)
        self.raw_materials: List[Product] = []
        self.max_employees = random.randint(self.config.MIN_COMPANY_SIZE, self.config.MAX_COMPANY_SIZE)
        self.bankruptcy = False
        self.profit_margin = 0.2  # (1 + profit margin) * price * sales = revenue  TODO: smart margin

    def __setattr__(self, key, value):
        if key == "funds" and hasattr(self, 'funds') and hasattr(self, 'agent_updates') and value > self.funds:
            self.update.attr_update(self,'revenue', value - self.funds)
        super().__setattr__(key, value)

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
    
    def get_all_unemployed(self):
        return [i for i in self.all_agents[Individual] if i.employer is None]

    def update_product_attributes(self, population: int, company_count: int):
        # Update product quality and price
        self.product.quality = max(1, self.calculate_product_quality())  # type: ignore
        self.product.price = (max(1, self.costs * (1 + self.profit_margin)) /  # type: ignore
                              self.estimate_sales(population=population, company_count=company_count))

    def hire_employee(self, candidate: Individual, salary: float) -> bool:
        #TODO: allow company to grow indefinitely
        if self.funds >= salary and len(self.employees) < self.max_employees:
            self.update.attr_update(candidate, 'employer', self)
            self.update.attr_update(candidate, 'salary', salary)
            self.update.attr_update(candidate, 'unemployed_state', 0)
            self.update.agent_list_update(self, 'employees', 'append', candidate)
            return True
        return False

    def fire_employee(self, employee: Individual):
        if employee in self.employees:
            self.employees.remove(employee)
            self.update.attr_update(employee, 'employer', None)
            self.update.attr_update(employee, 'salary', 0.0)

    # fix zero raw_material issue
    def find_raw_material(self, products: Iterable[Product]):
        # Get all products that does not have this company's product as a child material
        potential_materials = [p for p in products if self.product not in p.get_materials_recursive() and p is not self.product]

        if not potential_materials:
            return

        if self.product.quality == 1 or not self.raw_materials:
            self.update.agent_list_update(self, 'raw_materials', 'append', random.choice(potential_materials))
        if len(self.raw_materials) > 0 and random.random() < np.log(1 / len(self.raw_materials) + self.config.EPSILON) * 10:
            self.update.agent_list_update(self, 'raw_materials', 'append', random.choice(potential_materials))

    # fix forever growing raw_material issue
    def remove_raw_material(self):
        if len(self.raw_materials) > 1 and random.random() < 1 / np.log10(self.product.quality + self.config.EPSILON) / 10:
            self.update.agent_list_update(self, 'raw_materials', 'remove', random.choice(self.raw_materials))

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
        # self.all_agents.stats.num_bankruptcies += 1

    # Stage method
    def update_product(self) -> bool:
        self.update.clear_history()
        # Update company product prices and quality and reset revenue
        self.remove_raw_material()  # see if remove raw_material at this step
        self.update_product_attributes(population=len(self.all_agents[Individual]), company_count=len(self.all_agents[Company]))
        self.update.attr_update(self, 'revenue', 0)

        # see if able to find a raw_material at this step
        self.find_raw_material(ProductGroup(self.all_agents[Product]))
        return True

    # Stage method
    def do_finance(self) -> bool:
        self.update.clear_history()
        # Pay dividends from profit to owner
        self.transfer_funds_to(self.owner, self.dividend)
        # Check for bankruptcy
        if self.check_bankruptcy():
            self.declare_bankruptcy()
            # Add this company to a queue to be removed by the main thread
            self.update.remove_agent(self)
            return False

        # Pays employees salary
        [e.transfer_funds_from(self, e.salary) for e in self.employees]
        return True

    # Stage method
    def adjust_workforce(self) -> bool:
        self.update.clear_history()
        if self.revenue > self.costs * self.config.PROFIT_MARGIN_FOR_HIRING:
            # Hire new employees
            unemployed = self.get_all_unemployed()
            if unemployed:
                new_employee = max(unemployed, key=lambda x: x.talent)
                self.hire_employee(new_employee, new_employee.talent * self.config.SALARY_FACTOR)
        elif self.revenue < self.costs:
            # Fire employees
            if self.employees:
                employee_to_fire = random.choice(self.employees)
                self.fire_employee(employee_to_fire)
            # TODO: add a bankruptcy index every time when there's no worker to fire
        return True

    def report(self):
        return CompanyReports(
            step=self.step,
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