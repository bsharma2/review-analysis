function myFunc(vars) {
    return vars
}

function display_result() {
  // var loader = document.getElementById("#loader");
  // loader.style.display = "block";
  var x = document.getElementById("result");
  if (x.style.display === "none") {
    x.style.display = "block";
    var x = document.getElementById("myText").value;
    document.getElementById("entered_text").innerHTML = x;
    document.getElementById("entered_text").style.border = "thin ridge silver";
  } else {
    x.style.display = "none";
  }
}

function display_score_calculation() {
  var x = document.getElementById("score");
  if (x.style.display === "none") {
    x.style.display = "block";
  } else {
    x.style.display = "none";
  }
}

if(display) {

  // Rating number
  var rating_number = document.getElementById("rating_number");
  round_score = Math.round(score);
  var rating_emoji = "";
  for (i = 0; i < round_score; i++) {
    rating_emoji += String.fromCodePoint(0x1F31F);
  }
  rating_number.textContent = round_score + " " + rating_emoji;

  // Sentiment Value
  var semtiment_value = document.getElementById("sentiment_value");
  var sentiment_emoji = 0x1F61E;
  if(prediction == 'Positive') {
    sentiment_emoji = 0x1F60A;
  }
  semtiment_value.textContent = prediction + " " + String.fromCodePoint(sentiment_emoji);

  // Result
  var result = document.getElementById("result");
  result.style.display = "block";
  var visualizations_list = document.getElementById("visualizations");
  // Removes first elements [CLS]
  words.shift();
  weights.shift();
  // Removes last element [SEP]
  words.pop();
  weights.pop();  
  for (var idx in words) {
    var list_node = document.createElement("li");
    list_node.style.marginLeft = "10px";
    list_node.style.marginRight = "10px";
    var opacity = weights[idx] * 300
    if(prediction == 'Positive') {
      list_node.style.backgroundColor = 'rgba(30, 144, 255, ' + opacity + ')';
    }
    else {
      list_node.style.backgroundColor = 'rgba(30, 144, 255, ' + opacity + ')';
    }
    var textnode = document.createTextNode(words[idx]);
    list_node.appendChild(textnode)
    visualizations_list.append(list_node);
  }

  anychart.onDocumentReady(function() {

      var data = [
        {x: "5", value: rating_prediction_probability[4], normal: {fill: "#013220"}},
        {x: "4", value: rating_prediction_probability[3], normal: {fill: "#90EE90"}},
        {x: "3", value: rating_prediction_probability[2], normal: {fill: "#FFFF99"}},
        {x: "2", value: rating_prediction_probability[1], normal: {fill: "#ffcccb"}},
        {x: "1", value: rating_prediction_probability[0], normal: {fill: "#8b0000"}},
      ];

    console.log(data);


    // create the chart
    var chart = anychart.pie3d(data);

    // set the chart title
    chart.title("Detailed rating prediction");


    // draw
    chart.container("container");
    chart.draw();
  });
}




