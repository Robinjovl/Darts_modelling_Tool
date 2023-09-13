**1.0.4 [12-09-2023]**
- Small fixes.

**1.0.3 [11-09-2023]**
- Folders reorganized. 
- Breaking changes: physics creation changed:\
    **Before:**\  
        self.physics = Geothermal(...)\
    **Now:**\
        from darts.physics.geothermal.property_container import PropertyContainer\
        property_container = PropertyContainer()\
        self.physics = Geothermal(...)\
        self.physics.add_property_region(property_container)\
        self.physics.init_physics()


**1.0.2 [30-06-2023]**
- Wheels creation for Python 3.11 added.

**1.0.0 [16-06-2023]**
- Folders reorganized. 
- Breaking changes: import paths changed:\
    Before:\
        from darts.models.physics.geothermal import Geothermal\
        from darts.models.physics.iapws.iapws_property_vec import _Backward1_T_Ph_vec\
    Now:\
        from darts.physics.geothermal.physics import Geothermal\
        from darts.physics.properties.iapws.iapws_property_vec import _Backward1_T_Ph_vec\
- Stop wheels creation for Python 3.6

**0.1.4 [13-04-2023]**
- Added heat losses from wellbore. It works by default for all thermal models.
- Added capability for connection arbitrary well segments for modeling closed-loop wells. 
- Adedd CoaxWell model which models closed loop well with surrounded reservoir. 
- Added poromechanics tests.

**0.1.3 [06-03-2023]**
Initial release.
