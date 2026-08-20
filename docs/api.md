# API reference

Generated from the modules' own docstrings. The [concepts guide](guide.md)
covers how these fit together.

## Fields and layouts

### `heliostat.field`

::: heliostat.field
    options:
      members:
        - HeliostatField
        - load_field
        - downselect
        - coincident_pairs
        - neighbour_pairs

### `heliostat.field_layouts`

::: heliostat.field_layouts
    options:
      members:
        - generate
        - fermat_spiral
        - wedge_filter
        - ring_filter
        - road_corridors
        - min_spacing_filter
        - write_field_csv

## Geometry

### `heliostat.geometry.aperture`

::: heliostat.geometry.aperture
    options:
      members:
        - Region
        - Rect
        - Disc
        - Ellipse
        - Annulus
        - Polygon
        - regular_polygon
        - CircularArray
        - Translate
        - Rotate
        - Union
        - Intersection
        - Difference

### `heliostat.geometry.design`

::: heliostat.geometry.design
    options:
      members:
        - HeliostatDesign
        - Facet
        - Surface
        - Flat
        - Spherical
        - ZernikeAstig
        - rect_heliostat
        - grid_facets
        - flower
        - cant_on_axis

### `heliostat.geometry.aiming`

::: heliostat.geometry.aiming
    options:
      members:
        - Solution
        - solve_prime_focus
        - solve_axicon
        - solve_cassegrain
        - solve_pyramid
        - aim_points_mm

### `heliostat.geometry.secondary`

::: heliostat.geometry.secondary
    options:
      members:
        - Secondary
        - NoSecondary
        - AxiconSecondary
        - CassegrainSecondary
        - PyramidSecondary

### `heliostat.geometry.receiver`

::: heliostat.geometry.receiver
    options:
      members:
        - Receiver
        - FlatWindowReceiver
        - CylinderReceiver
        - FrustumReceiver

### `heliostat.geometry.shading`

::: heliostat.geometry.shading
    options:
      members:
        - MirrorGeometry
        - build_geometries
        - shading_blocking
        - occlusion_efficiency
        - polygon_occlusion
        - corner_shadow

## Tracing

### `heliostat.trace.modes`

::: heliostat.trace.modes
    options:
      members:
        - TraceMode

### `heliostat.trace.mc`

::: heliostat.trace.mc
    options:
      members:
        - trace_heliostat

### `heliostat.trace.cone`

::: heliostat.trace.cone
    options:
      members:
        - trace_heliostat_cone
        - sunshape_kernel

### `heliostat.trace.samplers`

::: heliostat.trace.samplers
    options:
      members:
        - SuperGaussSampler
        - BuieSampler
        - make_sampler

## Sun, weather and metrics

### `heliostat.solar`

::: heliostat.solar
    options:
      members:
        - sun_position
        - sunrise_sunset
        - declination_hour_angle
        - hours_of_year
        - build_time_grid
        - describe_time_grid
        - TimeStep

### `heliostat.dni`

::: heliostat.dni
    options:
      members:
        - DNIProvider
        - ConstantDNI
        - TableDNI
        - MonthlyProfileDNI
        - DailyClimatologyDNI
        - ClearSkyDNI
        - SolarTimeAligned
        - provider_for
        - load_dni_provider
        - fetch

### `heliostat.metrics`

::: heliostat.metrics
    options:
      members:
        - spot_metrics
        - map_metrics
        - aperture_metrics
        - encircled_energy
        - encircled_energy_rays
        - encircled_energy_radii
        - rank_heliostats

## Sweeps, storage and energy

### `heliostat.sweep`

::: heliostat.sweep
    options:
      members:
        - run_sweep
        - standard_optics
        - OpticsSpec

### `heliostat.store`

::: heliostat.store
    options:
      members:
        - RunStore
        - TimestepResult
        - flux_scale
        - scale_factor

### `heliostat.energy`

::: heliostat.energy
    options:
      members:
        - optical_efficiency
        - build_interpolator
        - annual_energy
        - traced_day_energy
        - per_heliostat_annual
        - cross_check_daily_energy
        - declination_coverage
        - suggest_sweep_dates
        - distinct_declinations
        - fit_annual_sine

## Command line and web app

### `heliostat.cli`

::: heliostat.cli
    options:
      members:
        - main

### `heliostat.web`

::: heliostat.web
    options:
      members:
        - create_app
