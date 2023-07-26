import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder #Encoding
from sklearn.linear_model import LinearRegression #Model
import pickle #Importing Pipeline
pipe = pd.read_pickle('used_car_price_finder.pkl')

pd.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings('ignore')

# Interface Building

st.set_page_config(page_title='The No Code ML App', page_icon='atom.png', layout="wide", initial_sidebar_state="auto", menu_items=None)

embed_component = {
    'Linkedin': """<script src="https://platform.linkedin.com/badges/js/profile.js" async defer type="text/javascript"></script>
    <div class="badge-base LI-profile-badge" data-locale="en_US" data-size="medium" data-theme="dark" data-type="VERTICAL" data-vanity="mayur-shrotriya" data-version="v1"><a class="badge-base__link LI-simple-link" href="https://in.linkedin.com/in/mayur-shrotriya?trk=profile-badge"></a></div>""" 
}

import streamlit.components.v1 as components
with st.sidebar:
    components.html(embed_component['Linkedin'], height=500)

###################################################### Aesthetics ######################################################################
hide_st_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """
st.markdown(hide_st_style, unsafe_allow_html=True)


c1 , c2, c3 , c4= st.columns([1,1,2,2])
# c2.image('icon.png', width=70)
c3.markdown("<h1 style='text-align: center;'><font face='High Tower Text'> Used Car Price Prediction </font></h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: right; color: #ffd11a;'><font face='Brush Script MT' weight=5 size=5>-By Mayur Shrotriya</font></p>", unsafe_allow_html=True)

st.markdown("***")
st.write()



# st.sidebar.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTKSiEsPKQAUEEk7kmEy1Rb7YotukM86O286A&usqp=CAU",width=150)
# st.sidebar.markdown("[Connect with me on Linkedin!](https://www.linkedin.com/in/mayur-shrotriya-b45133142/)")


is_new = st.radio('Is the car new?',['Yes', 'No'])
if is_new == 'Yes':
    is_new = 1
else:
    is_new = 0
    
franchise_make = st.selectbox('Make', ['Chevrolet', 'Jeep', 'Cadillac', 'Chrysler', 'Dodge', 'Kia', 'RAM',
                                                               'Mazda', 'Audi', 'Hyundai', 'Ford', 'Toyota', 'Lincoln', 'Volvo',
                                                               'GMC', 'Volkswagen', 'BMW', 'Lexus', 'Buick', 'Subaru', 'Scion',
                                                               'Honda', 'Acura', 'Nissan', 'INFINITI', 'Rolls-Royce', 'Bentley',
                                                               'Porsche', 'Mercedes-Benz', 'Jaguar', 'Land Rover', 'Maserati',
                                                               'Alfa Romeo', 'MINI', 'FIAT', 'Mitsubishi', 'Aston Martin',
                                                               'McLaren', 'Genesis', 'SRT', 'Lamborghini', 'Ferrari', 'Lotus',
                                                               'Shelby'])
model_name = st.selectbox('Model Name' , ['1 Series', '1500', '1M', '2 Series', '200', '3 Series', '3 Series Gran Turismo', '300', '3000GT', '300M', '4 Series', '4Runner', '5 Series', '5 Series Gran Turismo', '500', '500L', '500X', '6 Series', '6 Series Gran Turismo', '626', '7 Series', '8 Series', '86', '9-3', '9-3 SportCombi', '9-5', '9-7X', 'A-Class', 'A3', 'A4', 'A4 Allroad', 'A4 Avant', 'A5', 'A5 Sportback', 'A6', 'A6 Allroad', 'A7', 'A8', 'ATS', 'ATS Coupe', 'ATS-V', 'ATS-V Coupe', 'Acadia', 'Accent', 'Accord', 'Accord Coupe', 'Accord Crosstour', 'Accord Hybrid', 'ActiveHybrid 3', 'ActiveHybrid 7', 'Alero', 'Altima', 'Altima Coupe', 'Amanti', 'Armada', 'Arteon', 'Ascent', 'Aspen', 'Astra', 'Astro', 'Aura', 'Aurora', 'Avalanche', 'Avalon', 'Avenger', 'Aveo', 'Aviator', 'Azera', 'Aztek', 'B-Series', 'B9 Tribeca', 'BRZ', 'Baja', 'Beetle', 'Bentayga', 'Bentayga Hybrid', 'Blackwood', 'Blazer', 'Bonneville', 'Borrego', 'Bravada', 'C-Class', 'C-HR', 'C30', 'C70', 'CC', 'CL', 'CL-Class', 'CLA-Class', 'CLK-Class', 'CLS-Class', 'CR-V', 'CT4', 'CT5', 'CT6', 'CTS', 'CTS Coupe', 'CTS Sport Wagon', 'CTS-V', 'CTS-V Coupe', 'CTS-V Wagon', 'CX-3', 'CX-30', 'CX-5', 'CX-7', 'CX-9', 'Cabrio', 'Cadenza', 'Caliber', 'Camaro', 'Camry', 'Camry Solara', 'Canyon', 'Captiva Sport', 'Caravan', 'Cavalier', 'Cayenne', 'Celica', 'Century', 'Challenger', 'Charger', 'Cherokee', 'Civic', 'Civic Coupe', 'Civic Hatchback', 'Civic Hybrid', 'Civic Type R', 'Classic', 'Cobalt', 'Colorado', 'Commander', 'Compass', 'Concorde', 'Continental', 'Continental Flying Spur', 'Continental GT', 'Continental GTC', 'Continental Supersports', 'Cooper', 'Cooper Clubman', 'Cooper Paceman', 'Corolla', 'Corolla Hatchback', 'Corolla iM', 'Corsair', 'Countryman', 'Crosstour', 'Crosstrek', 'Crosstrek Hybrid', 'Crown Victoria', 'Cruze', 'Cruze Limited', 'Cube', 'DB9', 'DTS', 'Dakota', 'Dart', 'Dawn', 'DeVille', 'Defender', 'Diamante', 'Discovery', 'Discovery Sport', 'Durango', 'E-Class', 'E-PACE', 'E-Series', 'ECHO', 'ES', 'ES 300', 'ES 330', 'ES 350', 'EX35', 'Eclipse', 'Eclipse Spyder', 'EcoSport', 'Edge', 'Eighty-Eight Royale', 'Elantra', 'Elantra Coupe', 'Elantra GT', 'Elantra Touring', 'Eldorado', 'Element', 'Enclave', 'Encore', 'Encore GX', 'Endeavor', 'Entourage', 'Envision', 'Envoy', 'Envoy XL', 'Envoy XUV', 'Eos', 'Equator', 'Equinox', 'Equus', 'Escalade', 'Escalade ESV', 'Escalade EXT', 'Escape', 'Escape Hybrid', 'Escort', 'EuroVan', 'Expedition', 'Explorer', 'Explorer Hybrid', 'Explorer Sport', 'Explorer Sport Trac', 'Express', 'F-150', 'F-PACE', 'FJ Cruiser', 'FR-S', 'FX35', 'FX50', 'Fiesta', 'Firebird', 'Fit', 'Five Hundred', 'Fleetwood', 'Flex', 'Flying Spur', 'Focus', 'Focus RS', 'Forenza', 'Forester', 'Forte', 'Forte Koup', 'Forte5', 'Freestar', 'Freestyle', 'Frontier', 'Fusion', 'G-Class', 'G25', 'G3', 'G35', 'G37', 'G5', 'G6', 'G70', 'G8', 'G80', 'G90', 'GL-Class', 'GLA-Class', 'GLB-Class', 'GLC-Class', 'GLE-Class', 'GLK-Class', 'GLS-Class', 'GS', 'GS 200t', 'GS 300', 'GS 350', 'GS 430', 'GS F', 'GT-R', 'GTI', 'GTO', 'GX', 'GX 470', 'Galant', 'Genesis', 'Genesis Coupe', 'Ghibli', 'Ghost', 'Giulia', 'Gladiator', 'Golf', 'Golf Alltrack', 'Golf R', 'Golf SportWagen', 'GranTurismo', 'Grand Am', 'Grand Caravan', 'Grand Cherokee', 'Grand Marquis', 'Grand Prix', 'Grand Vitara', 'Grand Wagoneer', 'H3', 'H3T', 'HHR', 'HR-V', 'Highlander', 'I30', 'ILX', 'ILX Hybrid', 'ION', 'IPL G', 'IS', 'IS 250', 'IS 350', 'Impala', 'Impala Limited', 'Impreza', 'Impreza WRX', 'Impreza WRX STI', 'Intrepid', 'Intrigue', 'JX35', 'Jetta', 'Jetta GLI', 'Jetta Hybrid', 'Jetta SportWagen', 'Jimmy', 'Journey', 'Juke', 'K5', 'K900', 'Kicks', 'Kizashi', 'Kona', 'L-Series', 'L300', 'LC', 'LHS', 'LR2', 'LR3', 'LR4', 'LS', 'LS 400', 'LS 430', 'LS 460', 'LS 500', 'LX', 'LX 470', 'LX 570', 'LaCrosse', 'Lancer', 'Lancer Evolution', 'Lancer Sportback', 'Land Cruiser', 'LeSabre', 'Legacy', 'Liberty', 'Lucerne', 'Lumina', 'M-Class', 'M2', 'M3', 'M30', 'M35', 'M37', 'M4', 'M45', 'M5', 'M56', 'M6', 'M8', 'MAZDA2', 'MAZDA3', 'MAZDA5', 'MAZDA6', 'MAZDASPEED3', 'MAZDASPEED6', 'MDX', 'MKC', 'MKS', 'MKT', 'MKX', 'MKZ', 'MPV', 'Magnum', 'Malibu', 'Malibu Maxx', 'Mariner', 'Mariner Hybrid', 'Mark LT', 'Mark VIII', 'Matrix', 'Maxima', 'Metris', 'Milan', 'Mirage', 'Mirage G4', 'Montana', 'Montana SV6', 'Monte Carlo', 'Montego', 'Monterey', 'Montero', 'Montero Sport', 'Mountaineer', 'Mulsanne', 'Murano', 'Murano CrossCabriolet', 'Mustang', 'Mustang SVT Cobra', 'Mustang Shelby GT350', 'Mustang Shelby GT500', 'NX', 'NX 200t', 'Nautilus', 'Navigator', 'Neon', 'Neon SRT-4', 'Nitro', 'Odyssey', 'Optima', 'Optima Hybrid', 'Outback', 'Outlander', 'Outlander Sport', 'Outlook', 'PT Cruiser', 'Pacifica', 'Palisade', 'Panamera', 'Panamera Hybrid', 'Park Avenue', 'Passat', 'Passport', 'Pathfinder', 'Patriot', 'Phantom', 'Phantom Drophead Coupe', 'Pilot', 'Prelude', 'Prizm', 'ProMaster City', 'Protege5', 'Q3', 'Q40', 'Q5', 'Q50', 'Q60', 'Q7', 'Q70', 'Q70 Hybrid', 'Q70L', 'Q8', 'QX30', 'QX4', 'QX50', 'QX56', 'QX60', 'QX80', 'Quest', 'R-Class', 'R32', 'RAM 1500', 'RAV4', 'RC', 'RC 200t', 'RC 300', 'RC 350', 'RC F', 'RDX', 'RL', 'RLX Hybrid Sport', 'RS 3', 'RS 5', 'RS 5 Sportback', 'RS 7', 'RSX', 'RX', 'RX 300', 'RX 330', 'RX 350', 'RX-8', 'Rabbit', 'Raider', 'Rainier', 'Range Rover', 'Range Rover Evoque', 'Range Rover Hybrid', 'Range Rover Hybrid Plug-in', 'Range Rover Sport', 'Range Rover Velar', 'Ranger', 'Rapide', 'Regal', 'Regal Sportback', 'Regal TourX', 'Regency', 'Rendezvous', 'Renegade', 'Reno', 'Ridgeline', 'Rio', 'Rio5', 'Roadmaster', 'Rodeo', 'Rogue', 'Rogue Select', 'Rogue Sport', 'Rondo', 'Routan', 'S-10', 'S-Class', 'S-Class Coupe', 'S-Series', 'S-TYPE', 'S-TYPE R', 'S3', 'S4', 'S40', 'S5', 'S5 Sportback', 'S6', 'S60', 'S7', 'S8', 'S80', 'S90', 'SC 430', 'SQ5', 'SRX', 'SS', 'STS', 'SX4', 'Sable', 'Safari', 'Santa Fe', 'Santa Fe Sport', 'Santa Fe XL', 'Savana', 'Sebring', 'Sedona', 'Seltos', 'Sentra', 'Sequoia', 'Seville', 'Sienna', 'Sierra 1500', 'Sierra 1500 Limited', 'Sierra 1500HD', 'Sierra 2500HD', 'Sierra Classic 1500', 'Silhouette', 'Silverado 1500', 'Silverado 1500HD', 'Silverado Classic 1500', 'Silverado Classic 1500HD', 'Skylark', 'Sonata', 'Sonata Hybrid', 'Sonic', 'Sonoma', 'Sorento', 'Soul', 'Spark', 'Spectra', 'Sportage', 'Stealth', 'Stelvio', 'Stinger', 'Stratus', 'Suburban', 'Sunfire', 'Supra', 'TL', 'TLX', 'TSX', 'TT', 'TT RS', 'TTS', 'Tacoma', 'Tahoe', 'Taurus', 'Taurus X', 'Telluride', 'Terrain', 'Terraza', 'Tiburon', 'Tiguan', 'Titan', 'Torrent', 'Touareg', 'Touareg 2', 'Town & Country', 'Town Car', 'Tracker', 'Trailblazer', 'Trailblazer EXT', 'Transit Connect', 'Transit Crew', 'Transit Passenger', 'Traverse', 'Trax', 'Tribeca', 'Tribute', 'Trooper', 'Tucson', 'Tundra', 'UX', 'Uplander', 'V50', 'V60', 'V70', 'V90', 'VUE', 'Veloster', 'Veloster N', 'Veloster Turbo', 'Venture', 'Venue', 'Venza', 'Veracruz', 'Verano', 'Verona', 'Versa', 'Versa Note', 'Vibe', 'Vitara', 'Voyager', 'WRX', 'WRX STI', 'Windstar', 'Wraith', 'Wrangler', 'Wrangler Unlimited', 'X-TYPE', 'X1', 'X2', 'X3', 'X3 M', 'X4', 'X4 M', 'X5', 'X5 M', 'X6', 'X6 M', 'X7', 'XC40', 'XC60', 'XC70', 'XC90', 'XE', 'XF', 'XF Sportbrake', 'XG350', 'XJ-Series', 'XK-Series', 'XL-7', 'XT4', 'XT5', 'XT6', 'XTS', 'XV Crosstrek Hybrid', 'Xterra', 'Yaris', 'Yaris iA', 'Yukon', 'Yukon XL', 'ZDX', 'Zephyr', 'iA', 'iM', 'tC', 'xA', 'xB', 'xD'])
body_type    = st.selectbox('Select the Body Type of Vehicle', ['Sedan', 'Coupe', 'SUV / Crossover', 'Pickup Truck', 'Wagon', 'Minivan', 'Convertible', 'Hatchback', 'Van'])
color      = st.selectbox('Color', ['SILVER', 'BLACK', 'RED', 'WHITE', 'UNKNOWN', 'BLUE', 'GRAY', 'BROWN', 'YELLOW', 'ORANGE', 'GREEN', 'PURPLE', 'TEAL', 'GOLD', 'PINK'])


front_legroom = st.number_input('Front Legroom (in Inches)', max_value = 80)
back_legroom = st.number_input('Back Legroom (in Inches)', max_value = 80)


length       = st.number_input('Length of Vehicle', max_value=300)
width        = st.number_input('Width of Vehicle', max_value=120)
height       = st.number_input('Height of Vehicle', max_value=120)
max_seating  = st.number_input('Maximum no. of Seats',max_value=20)


city_fuel_economy = st.number_input('City Fuel Economy', max_value = 100)
highway_fuel_economy = st.number_input('Highway Fuel Economy', max_value = 100)
engine_displacement = st.number_input('Engine Displacement', max_value = 10000)
fuel_tank_volume = st.number_input('Fuel Tank Volume', max_value=60)
fuel_type = st.selectbox('Fuel Type', ['Gasoline', 'Flex Fuel Vehicle', 'Diesel', 'Hybrid', 'Biodiesel', 'Compressed Natural Gas'])

transmission = st.selectbox('Transmission Type', ['A', 'CVT' , 'M', 'Dual Clutch'])

transmission = 3 if transmission == 'A' else 2 if transmission == 'CVT' else 1 if transmission == 'M' else 0

transmission_display = st.selectbox('Transmission Display Type', ['6-Speed Automatic', '9-Speed Automatic', '8-Speed Automatic', '6-Speed Manual', '6-Speed Automatic Overdrive', '5-Speed Automatic', 'Continuously Variable Transmission', 'Automatic', '7-Speed Automatic', '4-Speed Automatic', 'Manual', '6-Speed Dual Clutch', '7-Speed Dual Clutch', '5-Speed Manual', '7-Speed CVT', '8-Speed CVT', '6-Speed CVT', '8-Speed Dual Clutch', '9-Speed Automatic Overdrive', '10-Speed Automatic', '8-Speed Manual', '8-Speed Automatic Overdrive', '5-Speed Automatic Overdrive', '5-Speed Manual Overdrive', '4-Speed Automatic Overdrive', '7-Speed Automatic Overdrive', '6-Speed Manual Overdrive', '3-Speed Automatic', '4-Speed Manual', '1-Speed CVT'])
i = transmission_display
f = ''
if i[0].isdigit():
    for j in i:
        if j.isdigit():
            f += j
elif i == 'Manual':
    f = 0
elif i == 'Continuously Variable Transmission':
    f = 1
elif i == 'Automatic':
    f = 2
transmission_display = f


wheel_system = st.selectbox('Wheel System', ['FWD', 'AWD', '4WD', 'RWD', '4X2'])
wheel_system = 2 if wheel_system == 'AWD' else 1 if wheel_system == '4WD' else 0


wheelbase = st.number_input('Wheel Base in Inches', max_value = 180)
horsepower = st.slider('HorsePower', max_value = 1000)
torque = st.slider('Torque (in lb/ft)', max_value = 1000)
no_of_cylinders = st.number_input('No. of Cylinders', max_value = 12)


vehicle_age = st.number_input('Age of the Vehicle (in Years)')
daysonmarket = st.number_input('Days on Market i.e. Days difference between today & Listing Date')

X = pd.DataFrame(
[[back_legroom, body_type, city_fuel_economy, daysonmarket, engine_displacement, franchise_make, front_legroom,
  fuel_tank_volume, fuel_type, height, highway_fuel_economy, horsepower, is_new, length, color, max_seating,
  model_name, transmission, transmission_display, wheel_system, wheelbase, width,torque,vehicle_age, no_of_cylinders]] , 
    columns = ['back_legroom_(inches)', 'body_type', 'city_fuel_economy', 'daysonmarket', 'engine_displacement', 'franchise_make', 'front_legroom_(inches)', 'fuel_tank_volume_(gallons)', 'fuel_type', 'height_(inches)', 'highway_fuel_economy', 'horsepower', 'is_new', 'length_(inches)', 'listing_color', 'maximum_seating', 'model_name', 'transmission', 'transmission_display', 'wheel_system', 'wheelbase', 'width_(inches)', 'Torque_lb_ft', 'Vehicle_Age', 'No_of_Cylinders'])

st.write(X.T.rename(columns={0:'Values')

price = pipe.predict(X)
prediction = "The Price of the car should be between "+ str(round(price[0]  + 500 ,2)) +' and '+ str(round(price[0] - 500, 2)) + ' USD'

if st.button('Predict Price of the Car'):
    st.success(prediction)
