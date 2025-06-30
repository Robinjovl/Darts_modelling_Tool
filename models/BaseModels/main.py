# from darts.engines import value_vector
from model import Model

m = Model(physics='do')
m.init()

m.set_output()

m.run(365)
m.print_timers()
# m.print_stat()
