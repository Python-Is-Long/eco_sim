from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar, Callable, Any, MutableMapping, Iterable, Optional
from abc import ABC, abstractmethod

from utils.data import to_db_types, convert_value
from utils.genericObjects import Agent

if TYPE_CHECKING:
    from utils.simulationObjects import Company, Individual, Product


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
    type: type['Agent']
    index: int

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


AgentType = TypeVar('AgentType', bound=Agent)
class SimulationAgents(dict):
    """A special dictionary type to store agents in the simulation."""
    def __init__(self, agents: Optional[Iterable[AgentType]] = None):
        super().__init__()
        if agents is None:
            agents = []
        for agent in agents:
            self.add(agent)

    def __getitem__(self, key: type[AgentType]) -> list[AgentType]:
        return super().__getitem__(key)

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
    args: tuple
    kwargs: dict

    def apply(self, agents: SimulationAgents):
        agent = self.agent_type(*self.args, **self.kwargs)
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
    updates: list[Update]

    def __init__(self, agents: SimulationAgents):
        from utils.simulationObjects import Company, Individual, Product
        self.agent_types = {
            Company: 'companies',
            Individual: 'individuals',
            Product: 'products',
        }
        self.agents = agents
        self.updates = []

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
        for u in self.updates:
            if isinstance(u, AttrUpdate) and u.agent_location == location and u.attr == attr:
                self.updates.remove(u)
                break
        self.updates.append(update)

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
        self.updates.append(update)

    def remove_agent(self, agent: 'Agent'):
        """Remove an agent from the simulation.

        Args:
            agent (Agent): The agent to remove.
        """
        location = self.locate_agent(agent)
        update = RemoveAgent(location)
        update.apply(self.agents)
        self.updates.append(update)

    def add_agent(self, agent_type: type['Agent'], *args, **kwargs):
        """Add an agent to the simulation.

        Args:
            agent_type: The type of agent to add.
            *args: The arguments for creating the agent.
            **kwargs: The keyword arguments for creating the agent.
        """
        update = AddAgent(agent_type, args, kwargs)
        update.apply(self.agents)
        self.updates.append(update)

    def clear(self):
        """Clear all updates."""
        self.updates.clear()
