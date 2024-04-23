const axios  = require('axios')
exports.home =(req,res)=>{
    res.send("This is home page");
}
exports.predict_parkinsons = async (req,res)=>{
    const result = await axios.post("http://localhost:5001/predictparkinsons",req.body);
    console.log(result.data)
    res.send("Fetched the data");
}