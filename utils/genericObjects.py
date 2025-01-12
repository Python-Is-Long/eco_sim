import uuid
import numpy as np
from typing import Union


def generate_name():
    """Returns a random uuid."""
    return uuid.uuid4()

class FundsObject():
    """A class to represent an object with funds.
    """
    def __init__(self, starting_funds: Union[int, float]=0, funds_precision: object=np.float64):
        self.funds_precision = funds_precision
        self.funds = funds_precision(starting_funds)

    @staticmethod
    def _warn_different_precision():
        print('Warning: Transferring funds between different precisions!')

    def set_funds(self, amount: Union[int, float]):
        """Set funds to a value."""
        self.funds = self.funds_precision(amount)
    
    def modify_funds(self, amount: Union[int, float]):
        """Modify funds by a value."""
        self.funds += self.funds_precision(amount)
    
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
        if self.funds_precision != object.funds_precision:
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
        if self.funds_precision != account.funds_precision:
            self._warn_different_precision()
        
        account.modify_funds(-amount)
        self.modify_funds(amount)
        return True


if __name__ == '__main__':
    funds1 = FundsObject(100)
    funds2 = FundsObject(50)

    print(funds1)
    print(funds2)
    funds1.transfer_funds_from(25, funds2)
    print(funds1)
    print(funds2)