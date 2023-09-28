from flask_wtf import FlaskForm
from wtforms import BooleanField, SubmitField, IntegerField, HiddenField, FloatField, FieldList, Form, FormField
from flask_wtf.file import FileField
from wtforms.validators import InputRequired


class ComponentForm(Form):
    component = FloatField(0, [InputRequired()])
    prefactor = FloatField(0, [InputRequired()])


class SearchForm(FlaskForm):

    message_field = HiddenField()
    output_data = HiddenField()
    file = FileField()
    file_name = HiddenField()
    generated_image = HiddenField()
    linear = FloatField(0, [InputRequired()])
    temp_power = IntegerField(validators=[InputRequired()])
    einstein_comps = FieldList(FormField(ComponentForm), min_entries=0)
    debye_comps = FieldList(FormField(ComponentForm), min_entries=0)
    optimise_values = BooleanField()
    log_x = BooleanField()
    log_y = BooleanField()
    start_T = FloatField(2, [InputRequired()])
    end_T = FloatField(300, [InputRequired()])
    submit = SubmitField("Calculate")
    submit
