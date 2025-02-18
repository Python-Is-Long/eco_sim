from abc import ABC, abstractmethod
from dataclasses import dataclass
import uuid
import numpy as np
from typing import Any, Iterable, TypeVar, Callable, Optional

from utils.data import convert_value, to_db_types


class Agent:
    """A generic agent class to store information about an agent in the simulation.
    Each agent should have access to the `SimulationAgents` object to interact with other agents.
    Agent's stats attributes should be updated through the `AgentUpdates` object.

    Attributes:
        name (str): The UUID for the agent.
        step (int): The current step of the simulation.
        update (AgentUpdates): An object to store updates to the agent.
        all_agents (SimulationAgents): An object to store all agents in the current simulation.
    """
    name: str
    step: int = 0
    update: 'AgentUpdates'
    all_agents: 'SimulationAgents'
    def __init__(self, all_agents: 'SimulationAgents'):
        """
        Initialize the agent with a UUID and the SimulationAgents object.
        
        Args:
            all_agents (SimulationAgents): An object to store all agents in the current simulation.
        """
        self.name = str(uuid.uuid4())
        self.all_agents = all_agents
        self.all_agents.add(self)
        self.update = AgentUpdates(all_agents)


class FundsAgent(Agent):
    """An agent that has funds and can transfer funds to other FundsAgent."""
    def __init__(self, all_agents: 'SimulationAgents', starting_funds: int|float = 0, funds_precision: Callable = np.float64):
        super().__init__(all_agents)

        self._funds_precision = funds_precision
        self.funds = starting_funds

    def __setattr__(self, key, value):
        if key == 'funds':
            value = self._funds_precision(value)
        super().__setattr__(key, value)

    @staticmethod
    def _warn_different_precision():
        print('Warning: Transferring funds between different precisions!')

    def set_funds(self, amount: int|float):
        """Set funds to a value."""
        self.update.attr_update(self, 'funds', amount)
    
    def modify_funds(self, amount: int|float):
        """Modify funds by a value."""
        self.update.attr_update(self, 'funds', self.funds + amount)
    
    def can_afford(self, amount: int|float) -> bool:
        """Check if current funds is above a specified amount."""
        return self.funds >= amount

    def transfer_funds_to(self, target_object: 'FundsAgent', amount: int|float) -> bool:
        """Transfer funds from current object to target object.
        Returns True if the transfer was successful, False otherwise.
        """
        if self.funds < amount:
            return False
        
        # Warning when transferring funds to an account with a different precision
        if self._funds_precision != target_object._funds_precision:
            self._warn_different_precision()
        
        self.modify_funds(-amount)
        target_object.modify_funds(amount)
        return True
    
    def transfer_funds_from(self, target_object: 'FundsAgent', amount: int|float) -> bool:
        """Transfer funds from target object to current object.
        Returns True if the transfer was successful, False otherwise.
        """
        if target_object.funds < amount:
            return False
        
        # Warning when transferring funds to an account with a different precision
        if self._funds_precision != target_object._funds_precision:
            self._warn_different_precision()
        
        target_object.modify_funds(-amount)
        self.modify_funds(amount)
        return True


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
        raise NotImplementedError("table_name method must be overridden in a subclass")

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
                converted_value = convert_value(current_value, field.agent_type)
            except Exception as e:
                raise TypeError(
                    f"Failed to convert field '{field.name}' from {type(current_value)} to {field.agent_type}: {e}"
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


@dataclass
class AgentLocation:
    """A dataclass to detonate the location of an agent in SimulationAgents."""
    type: type[Agent]
    index: int


class SimulationAgents(dict[type[Agent], list[Agent]]):
    """A special dictionary type to store agents in the simulation."""
    AgentType = TypeVar('AgentType', bound=Agent)
    def __init__(self, agents: Optional[Iterable[AgentType]] = None):
        super().__init__()
        if agents is None:
            agents = []
        for agent in agents:
            self.add(agent)

    def __getitem__(self, key: type[AgentType]) -> list[AgentType]:
        try:
            return super().__getitem__(key)  # type: ignore
        except KeyError:
            raise ValueError(f"Agent type {key.__name__} not found in the simulation")
    
    def __repr__(self) -> str:
        items_str = ", ".join(f"{key.__name__}: {value}" for key, value in self.items())
        return f"{{{items_str}}}"

    def add(self, agent: AgentType | Iterable[AgentType]):
        """Add a single agent or multiple agents to the simulation."""
        if isinstance(agent, Agent):
            agent = [agent]

        for a in agent:
            agent_list = self.get(type(a), [])
            if a in agent_list:
                raise ValueError(f"Agent {a} already in the simulation")
            agent_list.append(a)
            self[type(a)] = agent_list

    def remove(self, agent: AgentType | Iterable[AgentType]):
        """Remove a single agent or multiple agents from the simulation."""
        if isinstance(agent, Agent):
            agent = [agent]

        for a in agent:
            agent_list = self.get(type(a), [])
            if agent not in agent_list:
                raise ValueError(f"Agent {a} not in the simulation")
            agent_list.remove(a)

    def locate(self, location: AgentLocation) -> Agent:
        """Get the agent from given location in the simulation."""
        try:
            return self[location.type][location.index]
        except KeyError or IndexError:
            raise ValueError(f"Could not find agent at: {location.type}[{location.index}]")

    def get_location(self, agent: Agent) -> AgentLocation:
        for agent_type, agent_list in self.items():
            if agent in agent_list:
                return AgentLocation(agent_type, agent_list.index(agent))
        raise ValueError(f"Agent {agent} not found in the simulation")

    def step_increase(self):
        """Increase the step count of all agents in the simulation."""
        for agent_list in self.values():
            for agent in agent_list:
                agent.step += 1

    def clear_update_history(self):
        """Clear the update history of all agents in the simulation."""
        for agent_list in self.values():
            [agent.update.update_history.clear() for agent in agent_list]


class Update(ABC):
    """A dataclass to store updates to agents in the simulation."""
    @abstractmethod
    def apply(self, agents: dict[type['Agent'], list['Agent']]):
        """Apply this update to the simulation."""
        pass


@dataclass
class AttrUpdate(Update):
    """Modify an attribute of an agent. If the attribute is an agent type,
    is_agent should be set to True and the value should be the location of the agent.

    Attributes:
        agent_location: The location of the agent to modify.
        attr: The attribute to modify
        value: The new value of the attribute.
        is_agent: Whether the value is an Agent.
    """
    agent_location: AgentLocation
    attr: str
    value: Any
    is_agent: bool

    def apply(self, agents: dict[type['Agent'], list['Agent']]):
        if self.is_agent:
            agent = agents[self.value.type][self.value.index]
            setattr(agent, self.attr, self.value)
        else:
            setattr(self.agent_location, self.attr, self.value)


@dataclass
class AgentListUpdate(Update):
    """Modify an agent list attribute of an agent.

    Attributes:
        agent_location: The location of the agent to modify.
        attr: The attribute to modify
        mode: Either 'append' or 'remove'.
        modifying_agent: The location of the agent to append or remove.
    """
    agent_location: AgentLocation
    attr: str
    mode: str
    modifying_agent: AgentLocation

    def apply(self, agents: SimulationAgents):
        agent = agents[self.agent_location.type][self.agent_location.index]
        modifying_agent = agents[self.modifying_agent.type][self.modifying_agent.index]
        if self.mode == 'append':
            getattr(agent, self.attr).append(modifying_agent)
        elif self.mode == 'remove':
            getattr(agent, self.attr).remove(modifying_agent)


@dataclass
class AddAgent(Update):
    """Add an agent to the simulation.

    Attributes:
        agent_type: The type of agent to add.
        args: The arguments for creating the agent.
        kwargs: The keyword arguments for creating the agent.
    """
    agent_type: type['Agent']
    args: Iterable[Any]
    kwargs: dict[str, Any]

    def apply(self, agents: SimulationAgents):
        args = []
        for arg in self.args:
            if isinstance(arg, AgentLocation):
                args.append(agents.locate(arg))
        kwargs = {}
        for key, value in self.kwargs.items():
            if isinstance(value, AgentLocation):
                kwargs[key] = agents.locate(value)
            else:
                kwargs[key] = value

        agent = self.agent_type(*args, **kwargs)
        agents[self.agent_type].append(agent)


@dataclass
class RemoveAgent(Update):
    """Remove an agent from the simulation.

    Attributes:
        agent_location: The location of the agent to remove.
    """
    agent_location: AgentLocation
    def apply(self, agents: SimulationAgents):
        agents[self.agent_location.type].pop(self.agent_location.index)


class AgentUpdates:
    """A class to store agent updates in a simulation.
    This class is used to pass updates to agents though processes in the simulation.

    Attributes:
        updates: A dictionary with the agent location (agent type and index) as key and the function to update the agent as value.
    """
    update_history: list[Update]

    def __init__(self, agents: SimulationAgents):
        from utils.simulationObjects import Company, Individual, Product
        self.agent_types = {
            Company: 'companies',
            Individual: 'individuals',
            Product: 'products',
        }
        self.agents = agents
        self.update_history = []

    def locate_agent(self, agent: 'Agent') -> AgentLocation:
        """Locate an agent in the simulation.

        Returns:
            tuple: The agent type and index in the agent list.
        """
        agent_type = type(agent)
        try:
            return AgentLocation(agent_type, self.agents[agent_type].index(agent))
        except KeyError or ValueError:
            raise ValueError(f"Agent {agent} not found in any agent list")

    def attr_update(self, agent: 'Agent', attr: str, value: Any):
        """Update an attribute of an agent.

        Args:
            agent (Agent): The agent to update.
            attr (str): The attribute to update.
            value: The new value of the attribute.
        """
        location = self.locate_agent(agent)
        if not hasattr(agent, attr):
            raise ValueError(f"Agent {agent} does not have attribute {attr}")
        is_agent = isinstance(value, Agent)
        if is_agent:
            value = self.locate_agent(value)
        update = AttrUpdate(location, attr, value, is_agent)
        update.apply(self.agents)
        # Remove previous updates to the same attribute of the same agent if they exist
        for u in self.update_history:
            if isinstance(u, AttrUpdate) and u.agent_location == location and u.attr == attr:
                self.update_history.remove(u)
                break
        self.update_history.append(update)

    def agent_list_update(self, agent: 'Agent', attr: str, mode: str, modifying_agent: 'Agent'):
        """Update an agent list attribute of an agent. Either append or remove an agent from the list.

        Args:
            agent (Agent): The agent to update.
            attr (str): The attribute to update.
            mode (str): 'append' or 'remove'. The mode to update the list.
            modifying_agent (Agent): The agent to add or remove from the list.
        """
        if mode not in ['append', 'remove']:
            raise ValueError(f"Invalid mode '{mode}' for agent list update")

        attr_value = getattr(agent, attr)
        if not isinstance(attr_value, list):
            raise ValueError(f"Attribute {attr} of agent {agent} is not a list")

        if mode == 'remove' and not modifying_agent in attr_value:
            raise ValueError(f"Attempted to remove agent {modifying_agent} from list {attr} of agent {agent}, but agent not in list")

        location = self.locate_agent(agent)
        modifying_agent_location = self.locate_agent(modifying_agent)
        update = AgentListUpdate(location, attr, mode, modifying_agent_location)
        update.apply(self.agents)
        self.update_history.append(update)

    def add_agent(self, agent_type: type['Agent'], *args, **kwargs):
        """Add an agent to the simulation.

        Args:
            agent_type: The type of agent to add.
            *args: The arguments for creating the agent.
            **kwargs: The keyword arguments for creating the agent.
        """
        # Convert any agents types in the arguments to their locations to make the update easier to transfer between processes
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, Agent):
                args[i] = self.locate_agent(arg)
        for key, value in kwargs.items():
            if isinstance(value, Agent):
                kwargs[key] = self.locate_agent(value)

        update = AddAgent(agent_type, args, kwargs)
        update.apply(self.agents)
        self.update_history.append(update)

    def remove_agent(self, agent: 'Agent'):
        """Remove an agent from the simulation.

        Args:
            agent (Agent): The agent to remove.
        """
        location = self.locate_agent(agent)
        update = RemoveAgent(location)
        update.apply(self.agents)
        self.update_history.append(update)


if __name__ == '__main__':
    agents = SimulationAgents()
    funds1 = FundsAgent(agents, 100)
    funds2 = FundsAgent(agents, 50)
    print(agents)

    print(funds1.funds)
    print(funds2.funds)
    funds1.transfer_funds_from(funds2, 25)
    print(funds1.funds)
    print(funds2.funds)