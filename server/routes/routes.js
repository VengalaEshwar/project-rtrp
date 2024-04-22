const controller = require("../controllers/controller.js");
exports.route =(app)=>{
    app.get("/",controller.home);
}