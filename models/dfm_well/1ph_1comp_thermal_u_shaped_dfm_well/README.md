# Non-isothermal U-shaped DFM well

This example models a single, continuous DFM pipe with a descending leg, a
horizontal section, and an ascending leg. It uses the one-component thermal CO2
physics from `1ph_1comp_thermal_dfm_well_vs_olga`.

Injection is applied to segment 0 with `RampUpRate` and `set_rhs_flux`.
The  last segment has a volume of `1e8 m3`,  so its pressure changes negligibly over
the short example run and it acts as an approximate constant-pressure producer
boundary. Flow into that segment is the produced flow; the model does not
remove fluid from the large boundary volume.

The well has no reservoir perforations and no lateral heat exchange. The
`tang_2019` all-inclination drift-flux closure is used because the ascending leg
contains connections with inclination angles greater than 90 degrees.

## Sharp-elbow limitation

The current pipe model solves a scalar, one-dimensional momentum equation at
each connection. At a sharp change in direction, the connection inclination is
the arithmetic average of the two neighboring segment inclinations. Therefore,
the two 90-degree corners in this example are evaluated at 45 and 135 degrees,
respectively. This approximation affects the local gravity projection and the
inclination-dependent drift-flux closure.

The model does not include the vector momentum change, reaction force, bend
radius, or a local elbow-loss term such as `K rho v |v| / 2`. Consequently, this
example demonstrates continuous flow through a U-shaped pipe, but it should not
be used to predict pressure loss across sharp elbows without further model
development. For hydraulic studies sensitive to conditions near a bend, use a
smooth trajectory represented by several survey points and sufficient pipe
segments. A dedicated bend-loss term and a gravity projection based on the
actual elevation change between neighboring segment centroids should be added
when sharp elbows must be represented explicitly.
