# Imports
import os
import random
import traceback
from scipy.optimize import minimize, LinearConstraint
import string
import math
from flask import render_template, jsonify, make_response
from flask_restful import Resource, Api, reqparse
from math import e, exp
from jinja2 import BaseLoader, TemplateNotFound, ChoiceLoader
import logging
from logging.config import dictConfig
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.constants import R
from urllib import request, parse
import werkzeug
from werkzeug.utils import secure_filename
from app import app, api
from app.forms import SearchForm

parser = reqparse.RequestParser()
parser.add_argument('Debye_components')
parser.add_argument('Einstein_components')
parser.add_argument('linear_component')
parser.add_argument('start_temp')
parser.add_argument('end_temp')
parser.add_argument('exponent')
parser.add_argument('log_x')
parser.add_argument('log_y')
parser.add_argument("input_data")
parser.add_argument("data_file_name")

# As this is a backend must use different renderer
matplotlib.use('Agg')
logging.getLogger('matplotlib').setLevel(logging.ERROR)

app.config['UPLOAD_FOLDER'] = 'app/uploads'
app.config['GENERATED_IMAGES_FOLDER'] = 'app/static/generated_images'
app.config['GENERATED_DATA_FOLDER'] = 'app/static/generated_data'
app.config['MAX_CONTENT_PATH'] = 2e+6
char_set = string.ascii_uppercase


# We need both static and dynamic loader so we define something to choose
class UrlLoader(BaseLoader):
    def __init__(self, url_prefix):
        self.url_prefix = url_prefix

    def get_source(self, environment, template):
        url = parse.urljoin(self.url_prefix, template)
        try:
            t = request.urlopen(url)
            if t.getcode() is None or t.getcode() == 200:
                return t.read().decode('utf-8'), None, None
        except IOError:
            pass
        raise TemplateNotFound(template)


app.jinja_loader = ChoiceLoader([app.jinja_loader,
                                UrlLoader('https://lmds.liverpool.ac.uk/static/')])


dictConfig({"version": 1,
            "disable_existing_loggers": False,
            "formatters": {"default": {
                        "format": '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
                }},

            "handlers": {
                "wsgi": {
                    "class": "logging.StreamHandler",
                    "stream": "ext://flask.logging.wsgi_errors_stream",
                    "formatter": "default",
                    }
                },

            "root": {"level": "DEBUG", "handlers": ["wsgi"]},
            })


# Einstein Function
def Einstein(T, Te):
    Cve = ((3*R*(np.power((Te/T),2)))*((np.power(e,(Te/T)))/(np.power((np.power(e,(Te/T))-1),2))))
    return Cve


# Debye Function
def integrate(x):
    integ = ((np.power(x,4))*(exp(x)))/(np.power(exp(x)-1,2))
    return integ


def Debye(T, Td):
    Cvd = ((9*R*(np.power(T/Td,3)))*quad(integrate, 0, Td/T)[0])
    return Cvd


# Electrical Function
def y(gamma, T):
    j = (gamma*T)
    return j

#seperate linear, debye and einstein components from a single vector of parameters
def param_list_seperate(params, n_debye):
    
    debye_comps = [(params[i], params[i+1]) for i in range(0,n_debye*2,2)]
    assert len(debye_comps) == n_debye
    einstein_comps = []
    if len(params)%2 != 0:
        linear = params[-1]
        if n_debye * 2 + 1 < len(params):
            einstein_comps = [(params[i], params[i+1]) for i in range(n_debye*2,len(params)-2,2)]         
    else:
        linear = None
        if n_debye * 2 < len(params):
            einstein_comps = [(params[i], params[i+1]) for i in range(n_debye*2,len(params)-1,2)] 
    
    return debye_comps, einstein_comps, linear

def create_param_list(debye_comps, einstein_comps, linear):
    params = [p for pair in debye_comps for p in pair] 
    params += [p for pair in einstein_comps for p in pair]
    if linear is not None:
        params.append(linear)
    return params

def parameter_constrainer(debye_comps, einstein_comps, linear):
    total_prefactors = sum([math.fabs(x[1]) for x in debye_comps+einstein_comps])
    if linear is not None:
        total_prefactors += linear
        linear = math.fabs(linear/total_prefactors)
    einstein_comps = [(math.fabs(x[0]),math.fabs(x[1]/total_prefactors)) for x in einstein_comps]
    debye_comps = [(math.fabs(x[0]), math.fabs(x[1]/total_prefactors)) for x in debye_comps]
    return debye_comps, einstein_comps, linear



def model_heat_capacity(params, temps, true_values, n_debye, exponent_val, constrain_parameters=True,return_totals=False):
    # Extract parameters
    debye_comps, einstein_comps, linear = param_list_seperate(params, n_debye)
    if constrain_parameters:
        debye_comps, einstein_comps, linear = parameter_constrainer(debye_comps, einstein_comps, linear)
        
    # Update debye_comps, einstein_comps, and linear
    debye_ys = [[] for i in debye_comps]
    einstein_ys = [[] for i in einstein_comps]
    linear_ys = []
    # Model data
    for T in temps:
        for ys, (Td, Tdp) in zip(debye_ys, debye_comps):     
            if Tdp:
                ys.append((Tdp * Debye(T, Td))/np.power(T,exponent_val))

        for ys, (Te, Tep) in zip(einstein_ys, einstein_comps):
            if Tep:
                ys.append((Tep * Einstein(T, Te))/np.power(T,exponent_val))
        # Linear (gamma) part
        if linear is not None:
            linear_ys.append((y(linear, T))/np.power(T,exponent_val))
    all_ys = list(zip(*[x for x in debye_ys + einstein_ys + [linear_ys] if len(x) > 0]))
    
    totaly = list(map(sum, all_ys))
    if return_totals:
        return debye_ys, einstein_ys, linear_ys, totaly
    # Calculate the objective function as the sum of squared differences
    total_err = 0
    for T, yp, yhat in zip(temps,totaly,true_values):
        plot_y = yp/np.power(T,exponent_val)
        plot_yhat = yhat/np.power(T,exponent_val)
        total_err += math.fabs(plot_y-plot_yhat)/math.fabs(max(plot_y,plot_yhat))

    return total_err


def model_and_plot_heat_capacity(debye_comps, einstein_comps, linear, uploaded_file, start_t, end_t, log_x, log_y, exponent_val, optimise=False):
    data_0 = [float(x.split(',')[0].rstrip()) for x in uploaded_file]
    data_1 = [float(x.split(',')[1].rstrip()) for x in uploaded_file]
    
    #Temps to calculate
    temps = np.arange(start_t, end_t, 0.1)

    params = create_param_list(debye_comps, einstein_comps, linear)
    app.logger.debug(f"params {params}")
    if optimise:
        bounds = [(1.75,None) if i % 2 == 0 else (0,1) for i in range(len(params))]
        if linear is not None:
            bounds[-1] = (0,1)

        res = minimize(model_heat_capacity, params, args=(data_0, data_1, len(debye_comps), exponent_val), bounds=bounds)
        if not res.success:
            app.logger.debug("Failure to optimise")
            app.logger.debug(res.x)
            raise Exception(res.message)
        params = res.x
        app.logger.debug(f"params optimised: {params}")
        debye_comps, einstein_comps, linear = parameter_constrainer(*param_list_seperate(params, len(debye_comps)))
        params = create_param_list(debye_comps, einstein_comps, linear) # Now the param list is constrained
        
    debye_ys, einstein_ys, linear_ys, totaly = model_heat_capacity(params, temps, data_1, len(debye_comps), exponent_val, constrain_parameters=False,return_totals=True)

    # Plot
    plt.scatter(data_0,
                [data_1[i]/data_0[i]**int(exponent_val)
                    for i in range(len(data_0))],
                c="r",
                s=10,
                label="data")
    plt.plot(temps, totaly, c="black", label="total")
    data_dict = {"T": temps, "Total": totaly}
    for i, ys in enumerate(debye_ys):
        if len(ys) > 0:
            data_dict[f"D{i+1}"] = ys
            plt.plot(temps, ys, label=f"D{i+1}")
    for i, ys in enumerate(einstein_ys):
        if len(ys) > 0:
            data_dict[f"E{i+1}"] = ys
            plt.plot(temps, ys, label=f"E{i+1}")

    if linear is not None:
        data_dict["Linear"] = linear_ys
        plt.plot(temps, linear_ys, label="Linear")

    plt.xlabel("$T$")
    if exponent_val > 1:
        plt.ylabel(f"$C_p/T^{exponent_val}$")
    elif exponent_val == 1:
        plt.ylabel("$C_p/T$")
    elif exponent_val == 0:
        plt.ylabel("$C_p$")
    if log_x:
        plt.xlabel("$log T$")
        plt.xscale('log')
    if log_y:
        plt.yscale('log')
        if exponent_val > 1:
            plt.ylabel(f"$log(C_p/T^{exponent_val})$")
        elif exponent_val == 1:
            plt.ylabel("$log(C_p/T)$")
        elif exponent_val == 0:
            plt.ylabel("$log(C_p)$")

    plt.legend()
    image_file_name = ''.join(random.sample(char_set*6, 8))+'.png' 
    plt.tight_layout()
    # Save plot
    plt.savefig(os.path.join(app.config['GENERATED_IMAGES_FOLDER'], image_file_name))
    plt.close()

    # Direct user to saved plot
    data_file_name = ''.join(random.sample(char_set*6, 8))+'.csv'

    # Save resulting data
    df = pd.DataFrame(data_dict)
    df.to_csv(os.path.join(app.config['GENERATED_DATA_FOLDER'], data_file_name),index=False)
    if optimise:
        return image_file_name, data_file_name, debye_comps, einstein_comps, linear
    return image_file_name, data_file_name


@app.route("/", methods=['GET', 'POST'])
@app.route("/predict", methods=['GET', 'POST'])
def predict():
    form = SearchForm()
    form.output_data.data = None

    if form.validate_on_submit():

        try:
            # Process-file_here

            # If the user has uploaded a new file
            if form.file.data:
                file_name = ''.join(random.sample(char_set*6, 8))+'.cif'
                form.file.data.save(os.path.join(app.config['UPLOAD_FOLDER'], file_name))
                # Rather than having the user upload file every time,
                # we save it to temp folder, then record what we saved it as
                form.file_name.data = file_name

                with open(os.path.join(app.config['UPLOAD_FOLDER'],file_name), 'r') as f:
                    uploaded_file = f.readlines()
                # Remove file from form so user doesn't reupload every time
                form.file.data = None
            # If the user has previously uploaded file
            elif (form.file_name.data is not None) & (form.file_name.data != ""):
                with open(os.path.join(app.config['UPLOAD_FOLDER'],
                     secure_filename(form.file_name.data)), 'r') as f:
                    uploaded_file = f.readlines()

            # No file, no fun, return them to the form
            else:
                return render_template("heat_capacity.html", form=form)

            einstein_comps = [(x.component.data, x.prefactor.data)
                              for x in form.einstein_comps]
            app.logger.debug(einstein_comps)
            debye_comps = [(x.component.data, x.prefactor.data)
                           for x in form.debye_comps]
            app.logger.debug(debye_comps)
            # Check args
            if sum([x[1] for x in einstein_comps+debye_comps]) != 1:
                form.message_field.data = "The sum of the Debye and Einstein pre-factors do not sum to 1, they will be scaled accordingly in the amend this before downloading your model."
            if float(form.start_T.data) < 1.7:
                if form.message_field.data is None:
                    form.message_field.data = ""
                form.message_field.data += "<br> <b> Warning:</b> this model does not work below approximately 1.6 Kelvin"
            
            linear = form.linear.data if form.linear.data else None
            results = None
            if form.optimise_values.data:
                try:
                    image_name, out_data_name, debye_comps, einstein_comps, linear = model_and_plot_heat_capacity(debye_comps,
                                                einstein_comps, 
                                                linear, 
                                                uploaded_file,
                                                float(form.start_T.data), 
                                                float(form.end_T.data), 
                                                form.log_x.data, 
                                                form.log_y.data, 
                                                form.temp_power.data,
                                                optimise=form.optimise_values.data)
                    app.logger.debug("Updating values in form")
                    results = {"debye_comps":debye_comps, "einstein_comps":einstein_comps, "linear":linear}
                    for i in range(len(form.einstein_comps)):
                        form.einstein_comps[i].component.data = round(einstein_comps[i][0],3)
                        form.einstein_comps[i].prefactor.data = round(einstein_comps[i][1],3)
                    for i in range(len(form.debye_comps)):
                        form.debye_comps[i].component.data = round(debye_comps[i][0],3)
                        form.debye_comps[i].prefactor.data = round(debye_comps[i][1],3)
                    form.linear.data = round(linear,5)
                except: 
                    render_template("heat_capacity.html", form=form, message="Failed to calculate optimum, please adjust initial guesses")
            else:
                    image_name, out_data_name = model_and_plot_heat_capacity(debye_comps,
                                            einstein_comps, 
                                            linear, 
                                            uploaded_file,
                                            float(form.start_T.data), 
                                            float(form.end_T.data), 
                                            form.log_x.data, 
                                            form.log_y.data, 
                                            form.temp_power.data,
                                            optimise=form.optimise_values.data)
            

            form.output_data.data = os.path.join('heat_capacity/static/generated_data', out_data_name)
            form.generated_image.data = os.path.join('heat_capacity/static/generated_images',image_name)
            return render_template("heat_capacity.html", form=form, results=results)

        except Exception as e:
            app.logger.debug(e)
            app.logger.debug(traceback.print_exc())
    app.logger.debug("Validate on submit failed")
    form.generated_image.data = None
    form.output_data.data = None
    return render_template("heat_capacity.html", form=form)

@app.route("/API_info", methods=['GET', 'POST'])
def api_info():
    return render_template("api_info.html")

class ApiEndpoint(Resource):
    def get(self):
        return {"Hello": "world"}

    def put(self):
        args = parser.parse_args()
        if ("Debye_components" not in args) or ("Einstein_components" not in args) or \
           ("linear_component" not in args) or ("start_temp" not in args) or \
           ("end_temp" not in args) :
            return make_response(jsonify({"Error":"Failed to process input, check all required arguments are provided."}, 400))
        if ("data_file_name" not in args) and ("input_data" not in args):
            return make_response(jsonify({"Error":"No input data provided."}, 400))
        try:
            
            exponent_val = args["exponent"] if "exponent" in args else 1
            log_x = args["log_x"] if "log_x" in args else None
            log_y = args["log_y"] if "log_y" in args else None

            if "input_data" in args:
                file_name = ''.join(random.sample(char_set*6, 8))+'.cif'
                if isinstance(args["input_data"],str):
                    with open(os.path.join(app.config['UPLOAD_FOLDER'], file_name),"r") as f:
                        f.write(args["input_data"])
                else:
                    args["input_data"].save(os.path.join(app.config['UPLOAD_FOLDER'], file_name))


                with open(os.path.join(app.config['UPLOAD_FOLDER'],file_name), 'r') as f:
                    uploaded_file = f.readlines()
            else:
                with open(os.path.join(app.config['UPLOAD_FOLDER'],
                     secure_filename(args["data_file_name"])), 'r') as f:
                    uploaded_file = f.readlines()
            image_file_name, data_file_name = model_and_plot_heat_capacity(args["Debye_components"], args["Einstein_components"], args["linear_component"], uploaded_file,
                                                              args["start_temp"], args["end_temp"], log_x, log_y, exponent_val)

            return jsonify({"Heat Capacity Results": {"uploaded_file":file_name, 
            "image_file": os.path.join('heat_capacity/static/generated_images',image_file_name), 
            "data_file": os.path.join('heat_capacity/static/generated_data', data_file_name)}})
        except Exception as e:
            app.logger.debug(e)
            return make_response(jsonify({"Error":"Failed to process input, check it is properly formatted."}), 500)
    
    def post(self):
        args = parser.parse_args()
        return self.put()
api.add_resource(ApiEndpoint, "/API")
