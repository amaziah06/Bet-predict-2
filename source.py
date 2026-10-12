from .config import DEMO

if DEMO:
    from .demo import *  # noqa
else:
    from .apifootball import *  # noqa
