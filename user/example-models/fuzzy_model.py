import os
import tempfile

import numpy as np

from h1st.model.fuzzy import (
    FuzzyModeler, 
    FuzzyModel, 
    FuzzyMembership as fm,
    FuzzyVariables,
    FuzzyRules
)

def build_fuzzy_model():
    fuzzy_vars = FuzzyVariables()
    fuzzy_vars.add(
        var_name='var1',
        var_type='antecedent',
        var_range=np.arange(0, 10, 0.5),
        membership_funcs=[('normal', fm.GAUSSIAN, [3, 3.3]),
                          ('abnormal', fm.TRIANGLE, [8, 15, 15])]
    )
    fuzzy_vars.add(
        var_name='var2',
        var_type='antecedent',
        var_range=np.arange(0, 10, 0.5),
        membership_funcs=[('normal', fm.GAUSSIAN, [3, 3.3]),
                        ('abnormal', fm.TRIANGLE, [8, 15, 15])]
    )
    fuzzy_vars.add(
        var_name='conclusion1',
        var_type='consequent',
        var_range=np.arange(0, 10, 0.5),
        membership_funcs=[('no', fm.TRAPEZOID, [0, 0, 4, 6]),
                        ('yes', fm.TRAPEZOID, [4, 6, 10, 10])]
    )

    fuzzy_rule = FuzzyRules()
    fuzzy_rule.add(
        'rule1',
        if_=fuzzy_vars.var1['abnormal'] & fuzzy_vars.var2['abnormal'],
        then_=fuzzy_vars.conclusion1['yes']
    )
    fuzzy_rule.add(
        'rule2',
        if_=fuzzy_vars.var1['normal'],
        then_=fuzzy_vars.conclusion1['no']
    )
    fuzzy_rule.add(
        'rule3',
        if_=fuzzy_vars.var2['normal'],
        then_=fuzzy_vars.conclusion1['no']
    )

    modeler = FuzzyModeler()
    model = modeler.build_model(fuzzy_vars, fuzzy_rule)
    return model


if __name__ == "__main__":
    fuzzy_model = build_fuzzy_model()
    input_vars = {
            'var1': 7,
            'var2': 10
    }
    
    # Run prediction of Fuzzy Model.
    prediction = fuzzy_model.predict(input_vars)
    print("prediction['conclusion1']: ", prediction['conclusion1'])

    # Persist Fuzzy Model.
    with tempfile.TemporaryDirectory() as path:
        os.environ['H1ST_MODEL_REPO_PATH'] = path
        fuzzy_model.persist('my_version_1')

        # Load Fuzzy Model.
        reloaded_fuzzy_model = FuzzyModel().load('my_version_1')

    prediction = reloaded_fuzzy_model.predict(input_vars)
    print("prediction['conclusion1']: ", prediction['conclusion1'])

