from dataclasses import dataclass
import numpy as np
from typing import List, Optional
import random
import matplotlib.pyplot as plt
from tqdm import tqdm


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
        wealth_factor = np.tanh(self.wallet / 10000)  # Normalize wealth impact
        
        def score_product(product: Product) -> float:
            quality_weight = 0.5 + 0.5 * wealth_factor
            price_weight = 1.5 - 0.5 * wealth_factor
            return (quality_weight * product.quality - 
                   price_weight * product.price)
        
        scored_products = [(score_product(p), p) for p in products]
        if not scored_products:
            return None
        
        best_product = max(scored_products, key=lambda x: x[0])[1]
        if self.can_afford(best_product.price):
            return best_product
        return None

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
        
        # Base quality from employees
        employee_contribution = sum(emp.talent for emp in self.employees)
        diminishing_factor = np.log(len(self.employees) + 1)
        base_quality = employee_contribution * diminishing_factor
        
        # Additional quality from suppliers
        supplier_quality = sum(supplier.product_quality 
                             for supplier in self.suppliers)
        
        return base_quality + supplier_quality * 0.5

    def update_financials(self):
        total_salary = sum(emp.salary for emp in self.employees)
        supplier_costs = sum(supplier.product_price 
                           for supplier in self.suppliers)
        self.costs = total_salary + supplier_costs
        
        # Update product quality and price
        self.product_quality = self.calculate_product_quality()
        self.product_price = self.costs * 1.2  # 20% markup

    def hire_employee(self, candidate: Individual, salary: float) -> bool:
        if self.funds >= salary:
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
    
    def get_product(self) -> Product:
        """Return the product this company produces"""
        return Product(
            quality=self.product_quality,
            price=self.product_price,
            company=self
        )


class EconomyStats:
    def __init__(self):
        self.wealth_gini = []
        self.avg_wealth = []
        self.num_companies = []
        self.unemployment_rate = []
        self.avg_product_quality = []
        self.avg_product_price = []
        
    def calculate_gini(self, wealths: List[float]) -> float:
        """Calculate Gini coefficient to measure inequality"""
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

        # Print initial statistics
        employed = len([ind for ind in self.individuals if ind.employer])
        print(f"\nInitial Economy Statistics:")
        print(f"Total individuals: {len(self.individuals)}")
        print(f"Total companies: {len(self.companies)}")
        print(f"Employed individuals: {employed}")
        print(f"Unemployed individuals: {len(self.individuals) - employed}")
        print(f"Unemployment rate: {1 - employed/len(self.individuals):.2%}")
        
        # Print company sizes
        company_sizes = [len(company.employees) for company in self.companies]
        print(f"\nCompany size statistics:")
        print(f"Average company size: {np.mean(company_sizes):.1f}")
        print(f"Largest company: {max(company_sizes)}")
        print(f"Smallest company: {min(company_sizes)}")
        
    def _create_individuals(self, num_individuals: int) -> List[Individual]:
        talents = np.random.normal(100, 15, num_individuals)
        initial_funds = np.random.exponential(1000, num_individuals)
        return [Individual(t, f) for t, f in zip(talents, initial_funds)]
    
    def _create_companies(self, num_companies: int) -> List[Company]:
        companies = []
        available_workers = set(self.individuals)  # Create a set of available workers
        
        for _ in range(num_companies):
            # Choose owner from available workers
            owner = random.choice(list(available_workers))
            available_workers.remove(owner)  # Owner can't be an employee
            
            initial_funds = np.random.exponential(10000)
            company = Company(owner, initial_funds)
            
            # Randomly hire some initial employees
            num_initial_employees = min(
                company.max_employees,
                len(available_workers),
                random.randint(1, 10)  # Random initial workforce size
            )
            
            # Sort available workers by talent and randomly select from top 50%
            potential_employees = sorted(
                list(available_workers),
                key=lambda x: x.talent,
                reverse=True
            )[:int(len(available_workers) * 0.5)]
            
            if potential_employees:
                initial_employees = random.sample(
                    potential_employees,
                    min(num_initial_employees, len(potential_employees))
                )
                
                for employee in initial_employees:
                    # Initial salary based on talent
                    initial_salary = 50 + employee.talent * 0.5  # Basic salary formula
                    if company.hire_employee(employee, initial_salary):
                        available_workers.remove(employee)
            
            companies.append(company)
        
        print(f"Initial employment rate: {(len(self.individuals) - len(available_workers)) / len(self.individuals):.2%}")
        return companies
    
    def simulate_step(self):
        # Process company operations
        for company in self.companies:
            company.update_financials()
            
            # Pay salaries
            for employee in company.employees:
                if company.funds >= employee.salary:
                    company.funds -= employee.salary
                    employee.receive_money(employee.salary)
                else:
                    # Company is bankrupt
                    self.handle_bankruptcy(company)
                    break
        
        # Process individual spending
        for individual in self.individuals:
            # Probability of spending increases with wealth
            if random.random() < np.tanh(individual.wallet / 5000):
                # Get products from all companies
                available_products = [
                    company.get_product()
                    for company in self.companies
                ]
                chosen_product = individual.decide_purchase(available_products)
                if chosen_product:
                    if individual.spend(chosen_product.price):
                        chosen_product.company.funds += chosen_product.price
    
    def handle_bankruptcy(self, company: Company):
        for employee in company.employees[:]:
            company.fire_employee(employee)
        self.companies.remove(company)

    def collect_statistics(self):
        """Collect various economic indicators"""
        wealths = [ind.wallet for ind in self.individuals]
        employed = len([ind for ind in self.individuals if ind.employer])
        
        self.stats.wealth_gini.append(self.stats.calculate_gini(wealths))
        self.stats.avg_wealth.append(np.mean(wealths))
        self.stats.num_companies.append(len(self.companies))
        self.stats.unemployment_rate.append(
            1 - employed / len(self.individuals)
        )
        
        if self.companies:
            self.stats.avg_product_quality.append(
                np.mean([c.product_quality for c in self.companies])
            )
            self.stats.avg_product_price.append(
                np.mean([c.product_price for c in self.companies])
            )
        else:
            self.stats.avg_product_quality.append(0)
            self.stats.avg_product_price.append(0)

def run_simulation(
    num_individuals: int = 1000,
    num_companies: int = 50,
    num_steps: int = 1000
) -> Economy:
    """Run the economic simulation for specified number of steps"""
    
    # Initialize economy
    economy = Economy(num_individuals, num_companies)
    
    # Run simulation steps
    for _ in tqdm(range(num_steps), desc="Simulating economy"):
        economy.simulate_step()
        economy.collect_statistics()
    
    return economy

def plot_results(economy: Economy):
    """Plot various economic indicators"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Economic Simulation Results')
    
    # Wealth inequality (Gini coefficient)
    axes[0,0].plot(economy.stats.wealth_gini)
    axes[0,0].set_title('Wealth Inequality (Gini)')
    axes[0,0].set_ylabel('Gini Coefficient')
    
    # Average wealth
    axes[0,1].plot(economy.stats.avg_wealth)
    axes[0,1].set_title('Average Wealth')
    axes[0,1].set_ylabel('Wealth')
    
    # Number of companies
    axes[1,0].plot(economy.stats.num_companies)
    axes[1,0].set_title('Number of Companies')
    axes[1,0].set_ylabel('Count')
    
    # Unemployment rate
    axes[1,1].plot(economy.stats.unemployment_rate)
    axes[1,1].set_title('Unemployment Rate')
    axes[1,1].set_ylabel('Rate')
    
    # Average product quality
    axes[2,0].plot(economy.stats.avg_product_quality)
    axes[2,0].set_title('Average Product Quality')
    axes[2,0].set_ylabel('Quality')
    
    # Average product price
    axes[2,1].plot(economy.stats.avg_product_price)
    axes[2,1].set_title('Average Product Price')
    axes[2,1].set_ylabel('Price')
    
    plt.tight_layout()
    plt.show()

def print_summary(economy: Economy):
    """Print summary statistics"""
    print("\nFinal Economic Statistics:")
    print(f"Number of individuals: {len(economy.individuals)}")
    print(f"Number of companies: {len(economy.companies)}")
    print(f"Average wealth: {economy.stats.avg_wealth[-1]:.2f}")
    print(f"Wealth inequality (Gini): {economy.stats.wealth_gini[-1]:.2f}")
    print(f"Unemployment rate: {economy.stats.unemployment_rate[-1]:.2%}")
    
    # Wealth distribution
    wealth_percentiles = np.percentile(
        [ind.wallet for ind in economy.individuals],
        [25, 50, 75, 90, 99]
    )
    print("\nWealth Distribution:")
    print(f"25th percentile: {wealth_percentiles[0]:.2f}")
    print(f"Median: {wealth_percentiles[1]:.2f}")
    print(f"75th percentile: {wealth_percentiles[2]:.2f}")
    print(f"90th percentile: {wealth_percentiles[3]:.2f}")
    print(f"99th percentile: {wealth_percentiles[4]:.2f}")

# Run the simulation
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Run simulation
    economy = run_simulation(
        num_individuals=10000,
        num_companies=50,
        num_steps=1000
    )
    
    # Visualize results
    plot_results(economy)
    print_summary(economy)