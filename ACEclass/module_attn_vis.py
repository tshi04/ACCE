'''
@author Tian Shi
Please contact tshi@vt.edu
Based on Lin Zhouhan(@hantek) visualization codes.
'''
from codecs import open


def createHTML(input_data, fileName):
    """
    Creates a html file with text heat.
    weights: attention weights for visualizing
    texts: text on which attention weights are to be visualized
    """
    fOut = open(fileName, "w", encoding="utf-8")
    part1 = """
    <html lang="en">
    <head>
    <meta http-equiv="content-type" content="text/html; charset=utf-8">
    <style>
    body {
    font-family: Sans-Serif;
    }
    </style>
    </head>
    <body style="width: 400px">
    <h3>
    Heatmaps
    </h3>
    </body>
    <script>
    """
    part2 = """
    var color = "255,0,0";
    var weight_max = 0;
    for (var i = 0; i < myweights.length; i++) {
        if (myweights[i] > weight_max) {
            weight_max = myweights[i];
        }
    }

    var heat_text = "<span>gold label=" + gold_label + ", predicted label=" + pred_label + "</span><br>";
    heat_text += "<p style='border: 1px solid black; margin: 0; padding: 5px;'>";
    for (var i = 0; i < mytext.length; i++) {
        if (mytext[i].substring(0, 2) == "##") {
            heat_text += "<span style='background-color:rgba(" + color + "," + myweights[i]/weight_max + ")'>" + mytext[i].substring(2,) + "</span>";
        } else {
            if (i == 0) {
                heat_text += "<span style='background-color:rgba(" + color + "," + myweights[i]/weight_max + ")'>" + mytext[i] + "</span>";
            } else {
                heat_text += "<span style='background-color:rgba(" + color + "," + myweights[i]/weight_max + ")'>" + " " + mytext[i] + "</span>";
            }
        }
            
    }
    heat_text += "</p>";
    
    document.body.innerHTML += heat_text;
    
    </script>
    </html>"""
    input_text = []
    for wd in input_data['toks']:
        if wd == '"':
            wd = '\\"'
        input_text.append("\"{}\"".format(wd))
    input_text = ",".join(input_text)
    textsString = "var mytext = [{}];\n".format(input_text)
    input_weights = ",".join([str(wd) for wd in input_data['weights']])
    weightsString = "var myweights = [{}];\n".format(input_weights)

    gold_label = "var gold_label = {};\n".format(str(input_data['gold_label']))
    pred_label = "var pred_label = {};\n".format(str(input_data['pred_label']))

    fOut.write(part1)
    fOut.write(textsString)
    fOut.write(weightsString)
    fOut.write(gold_label)
    fOut.write(pred_label)
    fOut.write(part2)
    fOut.close()
