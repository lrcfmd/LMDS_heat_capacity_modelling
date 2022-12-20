# LMDS heat capacity modelling flask app

A git repository for the heat capacity modelling app on the LMDS.

This can be used as basis for more complex apps.
To deploy this app you need to add a scheduled job to your OS to regularly remove anything in the following folders:
* app/static/generated_data 
* app/static/generated_images
* app/uploads
