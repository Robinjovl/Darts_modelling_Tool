from darts.engines import timer_node

t = timer_node()
t.node['sds'] = timer_node()
t.node['sds'].get_timer()