"""Config overrides enabling the sources with global state dependences."""
import os

TORIC_NN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dummy_toric_nn.json')

SOURCES = {
    'cyclotron_radiation': {},
    'impurity_radiation': {
        'model_name': 'P_in_scaled_flat_profile',
        'fraction_P_heating': 0.1,
    },
    'icrh': {
        'model_path': TORIC_NN_PATH,
        'P_total': 10e6,
        'wall_inner': 1.24,
        'wall_outer': 2.43,
    },
}

# The ToricNN ICRH model needs the height of the magnetic axis, which the CHEASE
# geometry of the examples does not provide; the checks with the global sources
# therefore use a circular geometry.
GEOMETRY = {'geometry_type': 'circular'}
