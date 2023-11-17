
Engines
#######

.. automodule:: darts.engines
   :members: conn_mesh, engine_base, sim_params, ms_well, timer_node, pm_discretizer,
     engine_super_cpu1_1_t, engine_super_cpu2_1, engine_super_mp_cpu2_1, engine_super_elastic_cpu1_2,
     multilinear_adaptive_cpu_interpolator_i_d_1_1, multilinear_adaptive_cpu_interpolator_l_d_1_1,
     operator_set_evaluator_iface, operator_set_gradient_evaluator_iface, property_evaluator_iface
   :show-inheritance:

Physics
#######

.. automodule:: darts.physics
   :members: PhysicsBase, Geothermal, Compositional
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Geothermal
**********

.. autoclass:: darts.physics.geothermal.physics
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Super
*****

.. autoclass:: darts.physics.super.physics
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Properties
**********

.. autoclass:: darts.models.physics_sup.property_container.property_container
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Properties Black Oil
====================

.. automodule:: darts.models.physics_sup.properties_black_oil
   :members:
   :undoc-members:
   :show-inheritance: 
   :special-members: __init__

Properties CCS Thermal
======================

.. automodule:: darts.models.physics_sup.properties_ccs_thermal
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Basic Properties
================

.. automodule:: darts.models.physics_sup.properties_basic
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Operators
*********

.. automodule:: darts.models.physics_sup.operator_evaluator_sup
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Models
######

.. autoclass:: darts.models.darts_model.DartsModel
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Reservoirs
##########

.. autoclass:: darts.models.reservoirs.struct_reservoir.StructReservoir
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: darts.models.reservoirs.unstruct_reservoir.UnstructReservoir
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Discretizer
###########

Structured Discretizer
************************

.. automodule:: darts.mesh.struct_discretizer

.. autoclass:: StructDiscretizer
   :members:
   :special-members: __init__

Unstructured Discretizer
************************

.. automodule:: darts.mesh.unstruct_discretizer

.. autoclass:: UnstructDiscretizer
   :members:
   :special-members: __init__
