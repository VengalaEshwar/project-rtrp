const controller = require("../controllers/controller.js");
exports.route =(app)=>{
    app.get("/",controller.home);
    app.post("/predictparkinsons",controller.predict_parkinsons);
}