# NOTE: The previous module interface was generic of the implementation type.
# However, this currently seems overkill, as models never need to know the
# exact type of a specific (transfer) module. They only need access to the
# abstract `__call__` and `forward` functions. The actual type of the module is
# only important when instantiating, hence a non-generic abstract base for
# suffices for all module types.
