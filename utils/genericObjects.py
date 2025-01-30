import uuid
import numpy as np
from typing import Union, Optional, Any, Dict, List, Tuple


class NamedObject():
    """A class that assigns a unique name to each subclass."""
    def __init__(self):
        self.__init_subclass__()
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.name = uuid.uuid4()


class FundsObject():
    """A class that manages funds for a subclass."""
    def __init__(self, starting_funds: Union[int, float]=0, funds_precision: object=np.float64):
        self._funds_precision = funds_precision
        self.funds = funds_precision(starting_funds)
    
    def __init_subclass__(cls, starting_funds: Union[int, float]=0, funds_precision: object=np.float64, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._funds_precision = funds_precision
        cls.funds = funds_precision(starting_funds)

    @staticmethod
    def _warn_different_precision():
        print('Warning: Transferring funds between different precisions!')

    def set_funds(self, amount: Union[int, float]):
        """Set funds to a value."""
        self.funds = self._funds_precision(amount)
    
    def modify_funds(self, amount: Union[int, float]):
        """Modify funds by a value."""
        self.funds += self._funds_precision(amount)
    
    def can_afford(self, amount: Union[int, float]) -> bool:
        """Check if current funds is above a specified amount."""
        return self.funds >= amount

    def transfer_funds_to(self, object: 'type[FundsObject]', amount: Union[int, float]) -> bool:
        """Transfer funds from current object to a specified object.
        Returns True if the transfer was successful, False otherwise.
        """
        if self.funds < amount:
            return False
        
        # Warning when transferring funds to an account with a different precision
        if self._funds_precision != object._funds_precision:
            self._warn_different_precision()
        
        self.modify_funds(-amount)
        object.modify_funds(amount)
        return True
    
    def transfer_funds_from(self, account: 'type[FundsObject]', amount: Union[int, float]) -> bool:
        """Transfer funds from a specified object to current object.
        Returns True if the transfer was successful, False otherwise.
        """
        if account.funds < amount:
            return False
        
        # Warning when transferring funds to an account with a different precision
        if self._funds_precision != account._funds_precision:
            self._warn_different_precision()
        
        account.modify_funds(-amount)
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