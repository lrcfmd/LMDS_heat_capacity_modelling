# Imports
import os
import random
import traceback
import string
from flask import render_template
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
from werkzeug.utils import secure_filename
from app import app
from app.forms import SearchForm

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
    Cve = ((3*R*((Te/T)**2))*((e**(Te/T))/((e**(Te/T)-1)**2)))
    return Cve


# Debye Function
def integrate(x):
    integ = ((x**4)*(exp(x)))/(((exp(x)-1)**2))
    return integ


def Debye(T, Td):
    Cvd = ((9*R*((T/Td)**3))*quad(integrate, 0, Td/T)[0])
    return Cvd


# Electrical Function
def y(gamma, T):
    j = (gamma*T)
    return j


@app.route("/", methods=['GET', 'POST'])
@app.route("/predict", methods=['POST'])
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
            app.logger.debug("Power4:")
            app.logger.debug(form.temp_power.data)
            einstein_comps = [(x.component.data, x.prefactor.data)
                              for x in form.einstein_comps]
            app.logger.debug(einstein_comps)
            debye_comps = [(x.component.data, x.prefactor.data)
                           for x in form.debye_comps]
            app.logger.debug(debye_comps)
            # Check args
            if sum([x[1] for x in einstein_comps+debye_comps]) != 1:
                form.message_field.data = "Sum of proportions of Einstein and Debye components should be 1. Please adjust appropriately"
                return render_template("heat_capacity.html", form=form)
            app.logger.debug(uploaded_file)
            # Parse uploaded data
            data_0 = [float(x.split(',')[0].rstrip()) for x in uploaded_file]
            data_1 = [float(x.split(',')[1].rstrip()) for x in uploaded_file]

            # Create lists to fill with modeled data
            debye_ys = [[]] * len(debye_comps)
            einstein_ys = [[]] * len(einstein_comps)
            linear_ys = []
            # Temperatures at which to calculate
            temps = np.arange(float(form.start_T.data), float(form.end_T.data), 0.1)
            # Model data
            for T in temps:
                for ys, (Td, Tdp) in zip(debye_ys, debye_comps):
                    if Tdp:
                        ys.append((Tdp * Debye(T, Td))/T**form.temp_power.data)

                for ys, (Te, Tep) in zip(einstein_ys, einstein_comps):
                    if Tep:
                        ys.append((Tep * Einstein(T, Te))/T**form.temp_power.data)
                # Linear (gamma) part
                if form.linear.data:
                    linear_ys.append((y(form.linear.data, T))/T**form.temp_power.data)
            all_ys = zip(*[x for x in debye_ys + einstein_ys + [linear_ys] if len(x) > 0])
            totaly = list(map(sum, all_ys))

            # Plot
            plt.scatter(data_0,
                        [data_1[i]/data_0[i]**int(form.temp_power.data)
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

            if form.linear.data:
                data_dict["Linear"] = linear_ys
                plt.plot(temps, linear_ys, label="Linear")

            plt.xlabel("$T$")
            if form.temp_power.data > 1:
                plt.ylabel(f"$C_p/T^{form.temp_power.data}$")
            elif form.temp_power.data == 1:
                plt.ylabel("$C_p/T$")
            elif form.temp_power.data == 0:
                plt.ylabel("$C_p$")
            if form.log_x.data:
                plt.xlabel("$log T$")
                plt.xscale('log')
            if form.log_y.data:
                plt.yscale('log')
                if form.temp_power.data > 1:
                    plt.ylabel(f"$log(C_p/T^{form.temp_power.data})$")
                elif form.temp_power.data == 1:
                    plt.ylabel("$log(C_p/T)$")
                elif form.temp_power.data == 0:
                    plt.ylabel("$log(C_p)$")

            plt.legend()
            file_name = ''.join(random.sample(char_set*6, 8))+'.png'
            plt.tight_layout()
            # Save plot
            plt.savefig(os.path.join(app.config['GENERATED_IMAGES_FOLDER'],file_name))
            plt.close()

            # Direct user to saved plot
            form.generated_image.data = os.path.join('heat_capacity/static/generated_images',file_name)
            file_name = ''.join(random.sample(char_set*6, 8))+'.csv'

            # Save resulting data
            df = pd.DataFrame(data_dict)
            df.to_csv(os.path.join(app.config['GENERATED_DATA_FOLDER'],file_name),index=False)
            form.output_data.data = os.path.join('heat_capacity/static/generated_data', file_name)
            return render_template("heat_capacity.html", form=form)

        except Exception as e:
            app.logger.debug(e)
            app.logger.debug(traceback.print_exc())
    app.logger.debug("Validate on submit failed")
    return render_template("heat_capacity.html", form=form)
