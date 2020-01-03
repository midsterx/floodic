const express		= 	require("express")
const path			=	require("path")
const spawn			=	require("child_process").spawn
const bodyParser 	=	require('body-parser');

const app 			= 	express()

app.use(express.static(path.join(__dirname, 'assets')));
app.use(bodyParser.json()); // support json encoded bodies
app.use(bodyParser.urlencoded({ extended: true })); // support encoded bodies


app.get("/",function(req,res,err){
	res.sendFile("C://Users//sunri//Downloads//study material//sem 7//Research Project//MapUI//a.html")
})

app.get("/map",function(req,res,err){
	res.sendFile("C://Users//sunri//Downloads//study material//sem 7//Research Project//MapUI//map.html")
})
app.get("/map1",function(req,res,err){
	res.sendFile("C://Users//sunri//Downloads//study material//sem 7//Research Project//MapUI//map1.html")
})
app.post("/dateTime",function(req,res,err){
	console.log(req.body.date)
	console.log(req.body.time)
	console.log(req.body.date+" "+req.body.time)
	var process = spawn("C:\\Users\\sunri\\AppData\\Local\\Programs\\Python\\Python38\\python", ["C:/Users/sunri/Desktop/a.py",req.body.date+" "+req.body.time])
	process.stdout.on("data", function (data) {
    res.send(data.toString());
  });
	// res.send("helloooo")
})
app.listen(2000,function(){
	console.log("App is listening...")
})