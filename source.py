from .config import DEMO, PROVIDER

if DEMO:
    from .demo import *  # noqa
elif PROVIDER == "footballdata":
    from .footballdata import *  # noqa
else:
    from .apifootball import *  # noqa
