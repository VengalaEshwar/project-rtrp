const axios  = require('axios')
const cors = require('cors');
exports.home =(req,res)=>{
    res.send("This is home page");
}
exports.predict_parkinsons =  (req,res)=>{
    const result =  axios.post("http://localhost:5001/predictparkinsons",req.body);
    console.log(result.data);
    res.send("Fetched the data");
}