import uuid
import numpy as np
from typing import Union, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from utils.simulationUtils import AgentUpdates


class Agent:
    """A generic agent object that has a unique id for instance.

    Attributes:
        name(str): A UUID for the agent.
    """
    name: str
    step: int = 0
    update: 'AgentUpdates'
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = str(uuid.uuid4())


class FundsObject(Agent):
    """A class that manages funds for a subclass."""
    def __init__(self, starting_funds: Union[int, float]=0, funds_precision: Callable=np.float64, **kwargs):
        super().__init__(**kwargs)

        self._funds_precision = funds_precision
        self.funds = starting_funds

    def __setattr__(self, key, value):
        if key == 'funds':
            value = self._funds_precision(value)
        super().__setattr__(key, value)

    @staticmethod
    def _warn_different_precision():
        print('Warning: Transferring funds between different precisions!')

    def set_funds(self, amount: Union[int, float]):
        """Set funds to a value."""
        self.update.attr_update(self, 'funds', amount)
    
    def modify_funds(self, amount: Union[int, float]):
        """Modify funds by a value."""
        self.update.attr_update(self, 'funds', self.funds + amount)
    
    def can_afford(self, amount: Union[int, float]) -> bool:
        """Check if current funds is above a specified amount."""
        return self.funds >= amount

    def transfer_funds_to(self, target_object: 'FundsObject', amount: Union[int, float]) -> bool:
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
    
    def transfer_funds_from(self, target_object: 'FundsObject', amount: Union[int, float]) -> bool:
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
        

if __name__ == '__main__':
    funds1 = FundsObject(100)
    funds2 = FundsObject(50)

    print(funds1.funds)
    print(funds2.funds)
    funds1.transfer_funds_from(funds2, 25)
    print(funds1.funds)
    print(funds2.funds)