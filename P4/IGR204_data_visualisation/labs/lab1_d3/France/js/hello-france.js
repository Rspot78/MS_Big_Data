// alert("Hello, France!")


// ****************
// core program
// ****************


// user-specified variables
let dimensions = {"height": 700, "width":600}
let margins = {"top": 100, "bottom": 50, "left": 100, "right": 600}

// other variables automatically generated (do NOT change this)
let axes = {"vertical": dimensions.height + margins.top + 0.5 * margins.bottom, "horizontal": 0.5 * margins.left}; // axes locations
let buttons = {"horizontal": margins.left + dimensions.width + 100, "vertical1": margins.top , "vertical2": margins.top + 50}
let frame = {"horizontal": margins.left + dimensions.width + 100, "vertical": margins.top + 100}
let dataset = []; // list to store data later on
let show = true // boolean to show or not full information (true at start)

// create an SVG element "svg", which adds to the "body" part of html doc a "svg" tag with attributes "width" and "height"
let svg = d3.select("body")
    .append("svg")
    .attr("width", margins.left + dimensions.width + margins.right)
    .attr("height", margins.top + dimensions.height + margins.bottom);

// create a div element to add tooltips for cities
let div1 = d3.select("body")
    .append("div") 
    .attr("class", "tooltip")       
    .style("opacity", 0);

// create a second div element for full city information
let div2 = d3.select("body")
    .append("div") 
    .attr("class", "tooltip2")       
    .style("opacity", 0);

// create a button to allow display of full infirmation about cities
let button1 = d3.select("body")
    .append("button") 
    .attr("class", "button")
    .html("<b>Show full information</b>")
    .style("left", buttons.horizontal + "px")
    .style("top", buttons.vertical1 + "px")
    .attr("type", "radio")
    .on("click", callbackButton1)

// create a second button to hide full information about cities
let button2 = d3.select("body")
    .append("button") 
    .attr("class", "button")
    .html("<b>Hide full information</b>")
    .style("left", buttons.horizontal + "px")
    .style("top", buttons.vertical2 + "px")
    .attr("type", "radio")
    .on("click", callbackButton2)

// data loading (using the tsv mthod), then conversion of strings to doubles
d3.tsv("data/france.tsv")
    .row( (d, i) => {
        return {
            postalCode: +d["Postal Code"],
            inseeCode: +d.inseecode,
            place: d.place,
            longitude: +d.x,
            latitude: +d.y,
            population: +d.population,
            density: +d.density
        };
    })

    // display some log information from the console, and rescale the data to fit the canvas
    .get( (error, rows) => {
        console.log("Loaded " + rows.length + " rows");
        if (rows.length > 0) {
            console.log("First row: ", rows[0]);
            console.log("Last row: ", rows[rows.length-1]);
            x = d3.scaleLinear()
                .domain(d3.extent(rows, (row) => row.longitude))
                .range([margins.left, margins.left + dimensions.width]); // rescale data to fit canvas width                
            y = d3.scaleLinear()
                .domain(d3.extent(rows, (row) => row.latitude))
                .range([margins.top + dimensions.height, margins.top]); // rescale data to fit canvas height (note: latitude is reversed otherwise map is flipped!)                
            circle_scale = d3.scaleLinear()
                .domain(d3.extent(rows, (row) => row.population))
                .range([1, 50]); // rescale max circle size to match image proposed for the lab
            // circle_color = d3.scaleQuantile()
            circle_color = d3.scaleSqrt()
                .domain(d3.extent(rows, (row) => row.density))
                //.range(["rgb(100,100,150)", "rgb(150,100,200)"]) // scale colors: dark purple for low density, brighter purple for high densities
                .range(["rgb(100,100,150)", "rgb(200,150,250)"]) // scale colors: dark purple for low density, brighter purple for high densities
            dataset = rows; // fill "dataset" with data in "rows"
            draw(); // cast draw function to draw picture on canvas
        } 
    });



// **************
// functions    
// **************


// function to draw map, axes and title on canvas
function draw(){
    svg.selectAll("circle")  // basically says: rule for all circles :
        .data(dataset)
        .enter()  // when new data is added ...
        .append("circle")  // ... create a circle of size proportional to population ...
        .attr("r", (d) => circle_scale(d.population))
        .attr("cx", (d) => x(d.longitude))  // ... set its coordinate as longitude and latitude ...
        .attr("cy", (d) => y(d.latitude))
        .attr("opacity", 0.8)  // ... set its opacity to 0.8 ...
        .attr("fill", (d) => circle_color(d.density))  // ... and its color proportional to density.
        .on("mouseover", callbackMouseOver)
        .on("mouseout", callbackMouseOut)
    svg.append("g")  // create x axis
        .attr("class", "x axis")
        .attr("transform", "translate(0, " + axes.vertical + ")")
        .call(d3.axisBottom(x))
    svg.append("g")  // create y axis
        .attr("class", "y axis")
        .attr("transform", "translate(" + axes.horizontal + ",0)")
        .call(d3.axisLeft(y))
    svg.append("text")  // create title
        .attr("x", margins.left + dimensions.width / 2)             
        .attr("y", margins.top / 2)
        .attr("text-anchor", "middle")  
        .style("font-size", "42px") 
        .style("font-weight", 700)  
        .text("Lab 1: Hello, France (D3)");
}

// function on mouse passing over canvas
function callbackMouseOver(d,i){
    d3.select(this)
        .attr("fill", "rgb(200,100,200)") // change color to pinkish purple
        .attr("r", (d) => 4 + circle_scale(d.population));
    div1.transition()  // tooltip to display city names (note: inspired from https://bl.ocks.org/d3noob/a22c42db65eb00d4e369)
        .duration(20)    
        .style("opacity", .9);    
    div1.html("<b>City : </b>" + d.place + "<br/>" + "<b>Postal Code : </b>" + d.postalCode)  
        .style("left", (d3.event.pageX) + "px")   
        .style("top", (d3.event.pageY) + "px");
    if (show) { // allow tooltip display only if activated
        div2.transition()  // tooltip to display city names and postal code (note: inspired from https://bl.ocks.org/d3noob/a22c42db65eb00d4e369)
        .duration(20)    
        .style("opacity", .9);    
        div2.html("<b>City : </b>" + d.place + "<br/>" + "<br/>" + "<b>Postal Code : </b>" + d.postalCode + "<br/>" + "<br/>" + "<b> INSEE code: </b>" + d.inseeCode + "<br/>" + "<br/>" + "<b> population: </b>" + d.population + "<br/>" + "<br/>" + "<b>Density : </b>" + d.density + "<br/>" + "<br/>" + "<b>Longitude : </b>" + d.longitude + "<br/>" + "<br/>" + "<b>Latitude : </b>" + d.latitude  )  
        .style("left", frame.horizontal)   
        .style("top", frame.vertical);
    }
} 

// function on mouse leaving a circle on canvas
function callbackMouseOut(d,i){
    d3.select(this)
    .attr("fill", (d) => circle_color(d.density))
    .attr("r", (d) => circle_scale(d.population))
    div1.transition()    
    .duration(500)    
    .style("opacity", 0) 
}

// function on clicking button 1
function callbackButton1(d,i){
    show = true
}

// function on clicking button 2
function callbackButton2(d,i){
    div2.transition()    
    .duration(100)    
    .style("opacity", 0)
    show = false
}











































