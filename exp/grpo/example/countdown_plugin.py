import re
from typing import List

from swift.rewards import ORM, orms


class CountdownORM(ORM):

    def __call__(self, completions, target, nums, **kwargs) -> List[float]:
        rewards = []
        for completion, gt, numbers in zip(completions, target, nums):
            try:
                match = re.search(r'<answer>(.*?)</answer>', completion)
                if match is None:
                    rewards.append(0.0)
                    continue

                equation = match.group(1).strip().split('=', 1)[0]
                used_numbers = [int(n) for n in re.findall(r'\d+', equation)]
                if sorted(used_numbers) != sorted(numbers):
                    rewards.append(0.0)
                    continue
                if not re.match(r'^[\d+\-*/().\s]+$', equation):
                    rewards.append(0.0)
                    continue

                result = eval(equation, {'__builtins__': None}, {})
                rewards.append(float(abs(float(result) - float(gt)) < 1e-5))
            except Exception:
                rewards.append(0.0)
        return rewards


orms['external_countdown'] = CountdownORM
