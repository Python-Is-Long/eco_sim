from dataclasses import dataclass
import os
import pickle
import random
from typing import List, Dict, Optional, Sequence, Callable, Any
import multiprocessing as mp
from multiprocessing import shared_memory

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json

from utils.genericObjects import Agent
from utils.simulationObjects import Config, Individual, Company, ProductGroup
from utils.simulationObjects import IndividualReports, CompanyReports
from utils.database import DatabaseInfo, SimDatabase
from utils.simulationUtils import SimulationAgents, AgentLocation, Update


@dataclass
class Task:
    action: Callable[[Agent], Any]
    agent_location: AgentLocation
    shm_name: str


def tasking(task_queue: mp.Queue, return_queue: mp.Queue):
    current_shm_name = None
    while True:
        task: Task = task_queue.get()
        if task is None:
            break
        if current_shm_name != task.shm_name:
            shm = shared_memory.SharedMemory(name=task.shm_name)
            all_agents: SimulationAgents = pickle.loads(shm.buf[:])
            current_shm_name = task.shm_name
        agent = all_agents.locate(task.agent_location)
        task.action(agent)
        return_queue.put(agent.agent_updates.updates)


class Economy:
    step: int = 0
    creating_companies: List
    removing_companies: List[Company]
    staging: Sequence[tuple[type[Agent], Callable[..., bool]]]
    workers: List[mp.Process]

    def __init__(self, db_info: Optional[DatabaseInfo] = None, config: Config = Config()):
        self.config = config
        
        self.all_agents = SimulationAgents()
        self._create_individuals(self.config.NUM_INDIVIDUAL)
        self._create_companies(self.config.NUM_COMPANY)
        self.stats = EconomyStats()
        self.report_types = [IndividualReports, CompanyReports]
        self.creating_companies = []
        self.removing_companies = []
        self.threads = []
        self.multiprocess = False

        self.task_queue = mp.Queue()
        self.return_queue = mp.Queue()
        self.workers = []
        
        self.db_connection = True if db_info else False
        if db_info:
            self.db = SimDatabase(db_info, self.report_types)
            self.db.create_database('ECOSIM')

        self.staging = [
            (Company, Company.update_product),  # Update company product prices and quality and find new materials
            (Individual, Individual.purchase_product),  # Individuals spend money
            (Company, Company.do_finance),  # Company checks for bankruptcy and pays dividends then pays salaries
            (Individual, Individual.find_opportunities),  # Individuals find jobs or create new company
            (Company, Company.adjust_workforce),  # Adjust workforce for companies
        ]

    def _create_individuals(self, num_individuals: int):
        talents = np.random.normal(self.config.TALENT_MEAN, self.config.TALENT_STD, num_individuals)
        initial_funds = np.random.exponential(self.config.INITIAL_FUNDS_INDIVIDUAL, num_individuals)
        risk_tolerance = [round(random.uniform(0.5, 2.0), 2) for _ in range(num_individuals)]
        skills = [set(random.choices(self.config.POSSIBLE_MARKETS, k=self.config.MAX_SKILLS)) for _ in range(num_individuals)]
        [Individual(self.all_agents, t, f, skills=s, risk_tolerance=r, configuration=self.config) for t, f, s, r in zip(talents, initial_funds, skills, risk_tolerance)]

    def _create_companies(self, num_companies: int):
        available_workers = set(self.all_agents[Individual])

        for _ in range(num_companies):
            if not available_workers:
                break
            owner = random.choice(list(available_workers))
            available_workers.remove(owner)
            initial_funds = np.random.exponential(self.config.INITIAL_FUNDS_COMPANY)
            company = Company(self.all_agents, owner, initial_funds)

            # Hire initial employees
            num_initial_employees = min(company.max_employees, len(available_workers), random.randint(1, 10))
            potential_employees = sorted(list(available_workers), key=lambda x: x.talent, reverse=True)[
                                  :int(len(available_workers) * 0.5)]

            if potential_employees:
                initial_employees = random.sample(potential_employees,
                                                  min(num_initial_employees, len(potential_employees)))
                for employee in initial_employees:
                    initial_salary = 50 + employee.talent * 0.5
                    if company.hire_employee(employee, initial_salary):
                        available_workers.remove(employee)
    def start_processes(self):
        for _ in range(mp.cpu_count()):
            worker_process = mp.Process(target=tasking, args=(self.task_queue, self.return_queue))
            worker_process.start()
            self.workers.append(worker_process)
        self.multiprocess = True
    
    def end_processes(self):
        if not self.multiprocess:
            raise RuntimeError('Multiprocessing is not enabled!')

        for _ in range(len(self.workers)):
            self.task_queue.put(None)
        for worker in self.workers:
            worker.join()
        self.multiprocess = False

    def get_all_products(self) -> ProductGroup:
        return ProductGroup([company.product for company in self.all_agents[Company]])

    def get_all_unemployed(self):
        return [i for i in self.all_agents[Individual] if i.employer is None]
    
    def simulate_step(self):
        self.step += 1
        self.all_agents.step_increase()

        for company in self.all_agents[Company]:
            company.update_product()

        # Individuals spend money
        for individual in self.all_agents[Individual]:
            individual.purchase_product()

        # Company checks for bankruptcy and pays dividends then pays salaries
        for company in self.all_agents[Company]:
            company.do_finance()
        # self.handle_company_removal()

        # Individuals find jobs or create new company
        for individual in self.all_agents[Individual]:
            individual.find_opportunities()
        # self.handle_company_creation()

        # Adjust workforce for companies
        for company in self.all_agents[Company]:
            company.adjust_workforce()

        # Collect statistics
        self.collect_statistics()
        self.reports()

    def simulate_step_mp(self):
        if not self.multiprocess:
            print('Processes have not been started. Starting processes...')
            self.start_processes()
        self.step += 1
        self.all_agents.step_increase()
        
        for agent_type, action in self.staging:
            # Serialize the all agents and store them in a shared memory
            bytes_all_agents = pickle.dumps(self.all_agents)
            shm = shared_memory.SharedMemory(create=True, size=len(bytes_all_agents))
            shm.buf[:] = bytes_all_agents
            
            # Select the agents for the current stage and create tasks then send them to subprocesses
            stage_agent_count = len(self.all_agents[agent_type])
            for idx in range(stage_agent_count):
                task = Task(
                    action=action,
                    agent_location=AgentLocation(agent_type, idx),
                    shm_name=shm.name,
                )
                self.task_queue.put(task)
            
            # Process all the results returned by subprocesses
            for i in range(stage_agent_count):
                updates: list[Update] = self.return_queue.get()
                # Apply the updates returned by the subprocess in the main process
                [u.apply(self.all_agents) for u in updates]

            # Close the shared memory
            shm.close()
            shm.unlink()
            
            # Collect statistics
            self.collect_statistics()
            self.reports()

    def reports(self):
        if self.db_connection:
            self.db.insert_reports([i.report() for i in self.all_agents[Individual]])
            self.db.insert_reports([c.report() for c in self.all_agents[Company]])

    def save_state(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_state(filename: str) -> 'Economy':
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def collect_statistics(self):
        self.stats.step = self.step
        individual_wealths = [ind.funds for ind in self.all_agents[Individual]]
        company_wealths = [comp.funds for comp in self.all_agents[Company]]
        employed = len([ind for ind in self.all_agents[Individual] if ind.employer])

        self.stats.individual_wealth_gini.append(self.stats.calculate_gini(individual_wealths))
        self.stats.avg_individual_wealth.append(np.mean(individual_wealths))
        self.stats.sum_individual_wealth.append(np.sum(individual_wealths))
        self.stats.sum_company_wealth.append(np.sum(company_wealths))
        self.stats.num_companies.append(len(self.all_agents[Company]))
        self.stats.unemployment_rate.append(1 - employed / len(self.all_agents[Individual]))
        self.stats.bankruptcies_over_time.append(self.stats.num_bankruptcies)  # Track bankruptcies
        self.stats.new_companies_over_time.append(self.stats.num_new_companies)  # Track new companies

        self.stats.total_money = round(self.stats.sum_individual_wealth[-1] + self.stats.sum_company_wealth[-1])

        to_low_precision = lambda x: float(np.float32(x))
        self.stats.all_company_funds.append([to_low_precision(c.funds) for c in self.all_agents[Company]])
        self.stats.all_individual_funds.append([to_low_precision(i.funds) for i in self.all_agents[Individual]])
        self.stats.all_product_prices.append([to_low_precision(c.product.price) for c in self.all_agents[Company]])
        self.stats.all_salaries.append([to_low_precision(e.salary) for e in self.all_agents[Individual] if e.employer])
        self.stats.all_employee_counts.append([len(c.employees) for c in self.all_agents[Company]])

        if self.all_agents[Company]:
            self.stats.avg_product_quality.append(np.mean([c.product.quality for c in self.all_agents[Company]]))
            self.stats.avg_product_price.append(np.mean([c.product.price for c in self.all_agents[Company]]))
            self.stats.avg_company_raw_materials.append(np.mean([len(c.raw_materials) for c in self.all_agents[Company]]))
            self.stats.avg_company_employees.append(np.mean([len(c.employees) for c in self.all_agents[Company]]))
        else:
            self.stats.avg_product_quality.append(0)
            self.stats.avg_product_price.append(0)


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
        self.avg_company_raw_materials = []
        self.avg_company_employees = []
        self.all_employee_counts = []

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
                ['all_company_funds', 'all_individual_funds', 'all_product_prices', 'all_salaries', 'all_employee_counts']}

    @property
    def dict_time_series_attributes(self) -> Dict:
        return {attr: value for attr, value in self.__dict__.items() if
                isinstance(value, list) and attr not in self.dict_histogram_attributes}

    @staticmethod
    def calculate_gini(wealths: List[float]) -> float:
        sorted_wealths = sorted(wealths)
        n = len(sorted_wealths)
        if n == 0:
            return 0
        index = np.arange(1, n + 1)
        return ((2 * index - n - 1) * sorted_wealths).sum() / (n * sum(sorted_wealths))

    def save_stats(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)


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
        with open('db_info.json', 'r') as f:
            db_info = json.load(f)
        config = Config(
            NUM_INDIVIDUAL=num_individuals,
            NUM_COMPANY=num_companies,
        )
        # Note: Economy with database connection will not be picklable
        economy = Economy(
            # DatabaseInfo(**db_info),
            config=config,
        )
    for _ in tqdm(range(num_steps - economy.step), desc="Simulating economy"):
        # economy.simulate_step()
        economy.simulate_step_mp()
        # economy.all_companies[0].print_statistics()
        economy.stats.save_stats("simulation_stats.pkl")

        # Save simulation state
        # economy.save_state("economy_simulation.pkl")
    economy.end_processes()
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
    print(f"Number of individuals: {len(economy.all_agents[Individual])}")
    print(f"Number of companies: {len(economy.all_agents[Company])}")
    print(f"Average wealth: {economy.stats.avg_individual_wealth[-1]:.2f}")
    print(f"Wealth inequality (Gini): {economy.stats.individual_wealth_gini[-1]:.2f}")
    print(f"Unemployment rate: {economy.stats.unemployment_rate[-1]:.2%}")
    print(f"Total bankruptcies: {economy.stats.num_bankruptcies}")
    print(f"Total new companies: {economy.stats.num_new_companies}")

    wealth_percentiles = np.percentile([ind.funds for ind in economy.all_agents[Individual]], [25, 50, 75, 90, 99])
    print("\nWealth Distribution:")
    print(f"25th percentile: {wealth_percentiles[0]:.2f}")
    print(f"Median: {wealth_percentiles[1]:.2f}")
    print(f"75th percentile: {wealth_percentiles[2]:.2f}")
    print(f"90th percentile: {wealth_percentiles[3]:.2f}")
    print(f"99th percentile: {wealth_percentiles[4]:.2f}")


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    # Run simulation
    economy = run_simulation(
        num_individuals=1000,
        num_companies=500,
        num_steps=100,
        state_pickle_path="economy_simulation.pkl",
        resume_state=False,
    )

    # Load simulation state (optional)
    # economy = Economy.load_state("economy_simulation.pkl")

    # Plot results
    # plot_results(economy, save_path="economy_simulation_results.png")
    # print_summary(economy)