# Cut function Kernel
A novel kernel for rankings data.

# Installation instruction

### Python dependencies
- sympy
- GPFlow
- GPy
- networkx
- sklearn
- r2py
- pandas
- ipython
- tqdm

### Ubuntu dependencies
- libcurl4-openssl-dev
- build-essential 
- libcurl4-gnutls-dev 
- libxml2-dev 
- libssl-dev

### Installing r dependencies
```python
import rpy2
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector

base = importr('base')
utils = importr('utils')
packnames = ("kernlab", "combinat", "caret", "mvtnorm", "reshape2", "flexclust", "Rankcluster", "devtools")
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))
devtools = importr('devtools')
devtools.install_github("YunlongJiao/kernrank")
try:
    from rpy2.robjects.packages import importr
    import rpy2.robjects as ro
    kernrank = importr('kernrank')
except:
    raise ImportError("Make sure that rpy2 is installed and that kernrnak packgage is installed in R")
```





