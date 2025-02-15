import uuid
import numpy as np
from typing import Union, Callable


class NamedObject:
    """A class that assigns a unique name to each instance."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = uuid.uuid4()


class FundsObject(NamedObject):
    """A class that manages funds for a subclass."""
    def __init__(self, starting_funds: Union[int, float]=0, funds_precision: Callable=np.float64, **kwargs):
        super().__init__(**kwargs)

        self._funds_precision = funds_precision
        self.funds = funds_precision(starting_funds)

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

    def transfer_funds_to(self, target_object: 'FundsObject', amount: Union[int, float]) -> bool:
        if self.funds < amount:
            return False  # 沒有足夠的錢

        before_sender_funds = self.funds
        before_receiver_funds = target_object.funds

        # 進行資金轉移
        self.modify_funds(-amount)
        target_object.modify_funds(amount)

        after_sender_funds = self.funds
        after_receiver_funds = target_object.funds

        # 🚨 加入浮點數誤差修正
        total_before = before_sender_funds + before_receiver_funds
        total_after = after_sender_funds + after_receiver_funds

        if abs(total_before - total_after) > 1e-6:
            correction = total_before - total_after
            self.funds += correction / 2  # 讓誤差平分
            target_object.funds += correction / 2
            print(
                f"⚠️ 修正浮點數誤差: {correction:.10f}，現在 {self.name} = {self.funds:.10f}, {target_object.name} = {target_object.funds:.10f}")

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