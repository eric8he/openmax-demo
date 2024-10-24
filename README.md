openmax-demo
======
An end-to-end implementation of "Towards Open Set Deep Networks" (Bendale, Boult 2015). 
Existing implementations are not well-documented and are confusing to read, something that is fixed here.

## Files included

generate_arrs.py - lays groundwork 
openmax.py - the main event
templates/home.html - the frontend
static/styles/style.css - the frontend frosting
test-images/ - three images used to test in video/slideshow

## Running the project

To start the Flask webserver, simply run

	python openmax.py

Then the application will be available at http://localhost:8080/. If port 8080 is occupied, you can edit the SERV_PORT constant found at the top of the code.

This folder comes pre-loaded with a pre-calculated cifar10_dists.pickle file. If you wish to recalculate the MAVs and models for any reason, from the root directory, simply first run

    python generate_arrs.py

to generate the needed pickle file. This will take upwards of 10 minutes, depending on your computer specifications. Then remove the old cifar10_dists.pickle and rename array.pickle to cifar10_dists.pickle.
