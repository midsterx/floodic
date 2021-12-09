import sys
import gmplot
import random
import re


print(str("hellooooo 123"))
latitude_list = [12.919741,12.918160,12.918113,12.919731] 
longitude_list = [77.642876,77.641632,77.641707,77.642956] 
  
#gmap5 = gmplot.GoogleMapPlotter.from_geocode( "Outer Ring Road, beside Agara Lake Trail, HSR Layout" )  
gmap5 = gmplot.GoogleMapPlotter("12.918950","77.642100",19)  
#gmap5.scatter( latitude_list, longitude_list, '# FF0000', 
#                                size = 40, marker = False) 
gmap5.apikey = "" 
# polygon method Draw a polygon with 
# the help of coordinates 
print(sys.argv[1])
x=re.split(':|-| ',sys.argv[1])
x=[int(i) for i in x]
print(x)
print(sum(x))
random.seed(sum(x))
vec= random.choice(([1,0,0],[0,1,0],[0, 0, 1]))
#val = int(sys.argv[1])
if(vec[0]):
    gmap5.polygon(latitude_list, longitude_list, color = 'green') 
elif(vec[1]):
    gmap5.polygon(latitude_list, longitude_list, color = 'cornflowerblue') 
else:
    gmap5.polygon(latitude_list, longitude_list, color = 'red') 
  
gmap5.draw( "C:\\Users\\sunri\\Downloads\\study material\\sem 7\\Research Project\\MapUI\\map.html" ) 


sys.stdout.flush()
