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

.. autoclass:: darts.physics.physics_base.PhysicsBase
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Geothermal
***********

.. autoclass:: darts.physics.geothermal.physics.Geothermal
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Super
******

.. autoclass:: darts.physics.super.physics.Compositional
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Properties
***********

.. automodule:: darts.physics.super.property_container
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Properties Black Oil
====================

.. automodule:: darts.physics.properties.black_oil
   :members:
   :undoc-members:
   :show-inheritance: 
   :special-members: __init__

Basic Properties
=================

.. automodule:: darts.physics.properties.basic
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Conductivity Properties
========================

.. automodule:: darts.physics.properties.conductivity
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Enthalpy Properties
====================

.. automodule:: darts.physics.properties.enthalpy
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Density Properties
====================

.. automodule:: darts.physics.properties.density
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Viscosity Properties
=====================

.. automodule:: darts.physics.properties.viscosity
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Kinetics Properties
===================

.. automodule:: darts.physics.properties.kinetics
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Operators
*********

.. automodule:: darts.physics.super.operator_evaluator
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

.. autoclass:: darts.reservoirs.reservoir_base.ReservoirBase
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: darts.reservoirs.struct_reservoir.StructReservoir
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: darts.reservoirs.unstruct_reservoir.UnstructReservoir
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: darts.reservoirs.cpg_reservoir.CPG_Reservoir
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Mesh
****

Structured Discretizer
======================

.. automodule:: darts.reservoirs.mesh.struct_discretizer

.. autoclass:: StructDiscretizer
   :members:
   :special-members: __init__

Unstructured Discretizer
========================

.. automodule:: darts.reservoirs.mesh.unstruct_discretizer

.. autoclass:: UnstructDiscretizer
   :members:
   :special-members: __init__



