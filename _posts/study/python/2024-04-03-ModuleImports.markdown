---
layout: post
title:  Module Imports
date:   2024-04-03 19:15:16 +0900
categories: study
tags: python
---

<h4> Module Imports in .py files and .ipynb files </h4> 

Due to the difference in execution environment between .py files and .ipynb notebooks, module imports will behave differently.

While using an ipynb file as a testing ground for custom classes and functions defined in separate python files, I noticed this behaviour that even if I implement changes in the .py files, save the file, and re-run the import statements, the changes I made is not applied in the .ipynb files before restarting VS code IDE entirely. 

This is because of how the execution environment is handled. 

In jupyter notebooks, imported modules will be stored in a cache, in sys.modules. If import statements are re-run, python will actually just re-use the modules loaded in a cache instead of actually reimporting objects.  
This is because jupyter notebook maintains consistent kernel across cell operations. 

However, .py files will start new execution environment every time it is run anew. Therefore, only .py files will actually be able to reload the changed definitions in the modules being imported. 

You can use importlib.reload(moduleName) to explicitly reload a module of interest in a same kernel while using jupyter notebook. 

```python
import mymodule  # initially import your module
import importlib

# After editing mymodule.py, you can reload it using:
importlib.reload(mymodule)
```