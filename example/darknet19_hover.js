window.onload = function () {

  // var WIDTH = 1800;
  // var HEIGHT = 1200;
  var WIDTH = '7200px';
  var HEIGHT = '800px';
  var timer = 0;
  // ---- rotate dog image-----------------------------------------------
  dx = 0, dy = 0 + 60 * 3 * Math.sin(45 * Math.PI / 180);
  a = 1, b = 0, c = 0, d = 1;
  b = -Math.sin(30 * Math.PI / 180);
  b2 = -Math.sin(60 * Math.PI / 180);
  matrix = [a, b, c, d, dx, dy];

  // var x, y, s;

  // var zoom = d3.behavior.zoom()
  //   .translate([0, 0])
  //   .scaleExtent([1, 10])
  //   .scale(1)
  //   .on("zoom", zoomed);

  // function zoomed() {
  //   x = d3.event.translate[0];
  //   y = d3.event.translate[1];
  //   s = d3.event.scale;
  //   container.attr("transform", "translate(" + d3.event.translate + ") scale(" + d3.event.scale + ")");
  // }


  // var translate = 'translate(' + (WIDTH /5) + ',' + (0) + ')';
  var svg = d3.select("#darknet")
    .append("svg")
    .attr('id', 'darknet_svg')
    .attr('width', WIDTH)
    .attr('height', HEIGHT)
  // .call(zoom);
  // .call(d3.zoom().on("zoom", function () {
  //   svg.attr("transform", d3.event.transform)
  //   console.log(d3.event.transform)
  //   console.log("size:", d3.event.transform.k, ",x:", d3.event.transform.x, ",y:", d3.event.transform.y)
  // }))

  function image(imgName, indexX, indexY, w) {
    svg.append('g')
      .attr('transform', "matrix(" + [0.1, 0.3 * b2, 0, 0.3, indexX, indexY] + ")")
      .attr('x', 0)
      .attr('y', 0)
      .attr('width', 15)
      .attr('height', 15)
      .append('svg:image')
      .attr('class', 'image')
      .attr('width', w)
      .attr('height', w)
      // .attr('id', 'img_'+imgName.split(".",1))
      .attr('id', imgName)
      .attr('xlink:href', 'data/' + imgName)
      .style("opacity", 1);
    return indexX
  }

  function image_channel(imgName, indexX, indexY) {
    svg.append('g')
      .attr('transform', "matrix(" + [0.1, 0.3 * b2, 0, 0.3, indexX, indexY] + ")")
      .attr('x', 0)
      .attr('y', 0)
      .attr('width', 15)
      .attr('height', 15)
      .append('svg:image')
      .attr('class', 'image_channel')
      .attr('width', 490)
      .attr('height', 450)
      .attr('id', imgName)
      .attr('xlink:href', 'data/' + imgName)
      .style("opacity", 1);
    return indexX
  }

  function Filter(type, layer, id, deltaX, startx, starty, size, filterSize) {
    var conv = svg.append('g')
      // .attr('class', 'convImg')
      // .attr('id', 'conv'+layer+'_'+id)
      // .attr('transform', "matrix("+[size*0.1, size*0.3*b2, 0, size*0.3, startx, starty]+")" )
      .attr('x', 0)
      .attr('y', 0)
      .attr('width', 15)
      .attr('height', 15)
      .append('line')
      .attr('x1', startx + id / 2)
      .attr('y1', starty)
      .attr('x2', startx + id / 2)
      .attr('y2', starty + filterSize / 3.35)
      .attr("fill", "none")
      // .attr("stroke", "black")
      .attr("stroke", "#ffffff00")
      // .append('svg:image')          
      .attr('class', type + 'Img')
      .attr('id', 'layer_vis_l' + layer + '_f' + id)
    // .attr('xlink:href', './data/yolo_v2_generated/'+ 'layer_vis_l'+ layer +'_f' +id +'.jpg')          
    // .style("opacity", 0.8);     

    return size * 0.3 * b2;
  }

  function Convolution(type, layer, id, delta, startx, starty, size, filterSize) {
    var conv = svg.append('g')
      .attr('x', 0)
      .attr('y', 0)
      .attr('width', 15)
      .attr('height', 15)
      .append('line')
      .attr('x1', startx + id / 2 - 2)
      .attr('y1', starty)
      .attr('x2', startx + id / 2 + 2)
      .attr('y2', starty + filterSize / 3.35)
      .attr("fill", "none")
      // .attr("stroke", "black")
      .attr("stroke", "#ffffff00")
      // .append('svg:image')          
      .attr('class', type + 'Img')
      .attr('id', type + layer + '_' + id)

    // .attr('transform', "matrix(" + [size * 0.1, size * 0.3 * b2, 0, size * 0.3, startx, starty] + ")")
    // .attr('x', 0)
    // .attr('y', 0)
    // .attr('width', 15)
    // .attr('height', 15)
    // .append('svg:image')
    // .attr('class', type + 'Img')
    // .attr('id', type + layer + '_' + id)
    // .attr('xlink:href', './data/img/' + type + layer + '_' + id + '.jpg')
    // .style("opacity", 0.8);

    return size * 0.3 * b2;
  }

  //make line between CONVs 
  function makeLine(startX, endX, Y, id) {
    var lineData = [[startX, Y], [endX, Y]];
    var line = d3.line();
    svg.append("path")
      .attr('class', 'lines')
      .attr('id', 'line_' + id)
      .attr("d", line(lineData))
      .attr("stroke", "#55a5fa")
      .attr("stroke-width", "3px")
      .attr("fill", "none")
      .style("opacity", 0.5);


  }

  //make curve line between CONVs that is different row 
  function makeCurveLine(startX, startY, endX, endY, midY, id) {
    var p1 = [startX, startY],
      p2 = [startX + 10, startY],
      p3 = [startX + 10, midY],
      p4 = [endX - 10, midY],
      p5 = [endX - 10, endY],
      p6 = [endX, endY];

    var lineData = [p1, p2, p3, p4, p5, p6];
    var line = d3.line();

    svg.append("path")
      .attr('class', 'lines')
      .attr('id', 'line_' + id)
      .attr("d", line(lineData))
      .attr("stroke", "#55a5fa")
      .attr("stroke-width", "3px")
      .attr("fill", "none")
      .style("opacity", 0.5);

  }

  //make cube  
  function makeCube(startX, startY, endX, tensorNum, filterSize, color) {
    // console.log(startX, startY, endX, tensorNum, filterSize, color)
    var Size = filterSize / 416;
    var p1 = [startX, startY], //node 1,5
      p2 = [endX, startY], //node 2,8
      p3 = [endX, startY + 124 * Size], //node 3,9
      p4 = [startX, startY + 124 * Size], //node 4
      p5 = [startX + 41 * Size, startY - 108 * Size], //node 6
      p6 = [endX + 41 * Size, startY - 108 * Size], //node 7,11
      p7 = [endX + 41 * Size, startY + 16 * Size]; //node 10
    p8 = [startX + 41 * Size, startY + 16 * Size];

    var lineData = [
      p1, p2, p3, p4, p1,
      p5, p6, p2, p3, p7, p6,
      // p5, p8, p7, p3, p4, p8
    ];

    var line = d3.line();

    var cube = svg.append("path")
      .attr('class', 'cubes')
      .attr('id', 'cube')
      .attr("d", line(lineData))
      // .attr("stroke", "#55a5fa")
      .attr("stroke", color)
      .attr("stroke-width", "1.1px")
      .attr("fill", "none")
      .attr("z-index", -50)
      .style("opacity", 0.8);

    var lineData_light = [
      p5, p8, p7, p8, p4
    ];
    var cube_light = svg.append("path")
      .attr('class', 'cubes')
      .attr('id', 'cube')
      .attr("d", line(lineData_light))
      .attr("stroke", color + '63')
      .attr("stroke-width", "1px")
      .attr("fill", "none")
      .attr("z-index", -50)
      .style("opacity", 0.7);

    // console.log(startX, startY, endX, tensorNum, filterSize, color);
    var cubeText = [p4, p1, p5, p6];
    textOnCube(svg, cubeText, tensorNum, filterSize);

    // d3.select('#lines').on('click', function() {
    // });
  }
  //---------------------------------------------------------------
  function makeLayer(layer, Type, tensorNum, startX, startY, filterSize) {
    var endX = 0;
    var Size = 1;
    var color = 'blue'
    var layerType = ''

    // tensorNum = tensorNum > 256 ? tensorNum*0.8 : tensorNum;

    if (Type == 'Conv') {
      layerType = 'conv'
      for (var i = 0; i < tensorNum; i++) {
        Convolution(layerType, layer, i, startX + i * 0.5, startX, startY, Size, filterSize);
        endX = startX + i * 0.5;
        color = '#55a5fa';
      }
    } else if (Type == 'Maxpool') {
      layerType = 'maxpool'
      for (var i = 0; i < tensorNum; i++) {
        Convolution(layerType, layer, i, startX + i * 0.5, startX, startY, Size, filterSize);
        endX = startX + i * 0.5;
        color = '#ff1111';
      }
    } else if (Type == 'Filter') {
      layerType = 'filter'
      for (var i = 0; i < tensorNum; i++) {
        Filter(layerType, layer, i, startX + i * 0.5, startX, startY, Size, filterSize);
        endX = startX + i * 0.5;
        color = '#ff6611';
      }
    } else if (Type == 'route') {
      // console.log('layer:',layer);
    }
    // console.log(startX, startY, endX, tensorNum, filterSize);
    makeCube(startX, startY, endX, tensorNum, filterSize, color);
    return endX;
  }
  //---------------------------------------------------------------
  function textOnCube(svg, line, tensorNum, filterSize) {
    svg.append("text")
      .attr("x", (line[0][0] + line[1][0]) / 2)
      .attr("y", (line[0][1] + line[1][1]) / 2)
      .attr("dx", "-1.5em")
      .style("font-size", "8px")
      .text(filterSize);

    svg.append("text")
      .attr("x", (line[1][0] + line[2][0]) / 2)
      .attr("y", (line[1][1] + line[2][1]) / 2)
      .attr("dx", "-1.5em")
      .style("font-size", "8px")
      .text(filterSize);

    svg.append("text")
      .attr("x", (line[2][0] + line[3][0]) / 2)
      .attr("y", (line[2][1] + line[3][1]) / 2)
      .attr("dx", "-1.5em")
      .style("font-size", "8px")
      .text(tensorNum);

  }
  //---------------------------------------------------------------
  // function imageClick(){

  //   d3.selectAll(".maxpoolImg")
  //     .transition() // First fade to green.
  //       .delay(1000)
  //       .style("opacity", 0.2)


  //   d3.selectAll(".maxpoolImg")
  //     .transition() // First fade to green.
  //       .delay(5000)
  //       .style("opacity", 0.2)
  //       .remove();

  // }


  //---------------------------------------------------------------

  function imageHover() {

    $("image.image").hover(
      function () {
        $("#imgscale").css("display", "block").css("position", "absolute").css("top", (event.offsetY + 10) + "px").css("left", (event.offsetX + 10) + "px");
        $("#imgscale").append("<img src='" + 'data/' + this.id + "'style='width:200px'>");
        $("#imgscale").append("<p>" + this.id + "</p>");
      }, function () {
        $("#imgscale").find("img").remove();
        $("#imgscale").find("p").remove();
      }
    );

    $(".convImg").hover(
      function () {
        // var data = this.id.split(".", 1)
        // var layer = this.id.split(".", 1)
        // var num = this.id.split(".", 2)

        $("#imgscale").css("display", "block").css("position", "absolute").css("top", (event.offsetY + 10) + "px").css("left", (event.offsetX - 10) + "px").css("width", "300px");
        $("#imgscale").append("<img src='" + 'data/img/' + this.id + '.jpg' + "'style='width:200px'>");
        $("#imgscale").append("<p>" + this.id + '.jpg' + "</p>");
      }, function () {
        $("#imgscale").find("img").remove();
        $("#imgscale").find("p").remove();
      }
    );

    $(".maxpoolImg").hover(
      function () {
        $("#imgscale").css("display", "block").css("position", "absolute").css("top", (event.offsetY + 10) + "px").css("left", (event.offsetX - 10) + "px");
        $("#imgscale").append("<img src='" + 'data/img/' + this.id + '.jpg' + "'style='width:200px'>");
        $("#imgscale").append("<p>" + this.id + '.jpg' + "</p>");
      }, function () {
        $("#imgscale").find("img").remove();
        $("#imgscale").find("p").remove();
      }
    );

    $(".filterImg").hover(
      function () {
        console.log(this.id);
        var id = this.id;
        var layer = id.split("layer_vis", 2);
        console.log(layer[1]);

        $("#imgscale").css("display", "block").css("position", "absolute").css("top", (event.offsetY + 10) + "px").css("left", (event.offsetX - 10) + "px");
        // $("#imgscale").append("<img src='" + 'data/yolo_v2_generated/' + this.id + '.jpg' + "'style='width:200px'>");
        // $("#imgscale").append("<p>" + this.id + '.jpg' + "</p>");
        $("#imgscale").append("<img src='" + 'data/generated/' + 'input' + layer[1] + '.jpg' + "'style='width:200px'>");
        // $("#imgscale").append("<p>" + 'input' + layer[1] + '.jpg' + "</p>");
        $("#imgscale").append("<p>" + 'input' + "</p>");

        $("#imgscale").css("display", "block").css("position", "absolute").css("top", (event.offsetY + 10) + "px").css("left", (event.offsetX - 10) + "px");
        $("#imgscale").append("<img src='" + 'data/generated/' + 'output' + layer[1] + '.jpg' + "'style='width:200px'>");
        // $("#imgscale").append("<p>" + 'output' + layer[1] + '.jpg' + "</p>");
        $("#imgscale").append("<p>" + 'output' + "</p>");
      }, function () {
        $("#imgscale").find("img").remove();
        $("#imgscale").find("p").remove();
      }
    );

  }


  function darknet_19() {
    var end2 = 0;
    var end = image('input.jpg', 20, 350, 500);
    makeLine(end + 55, end + 75, 350, 1);
    makeLayer(-1, 'Conv', 3, 88, 350, 416);
    image('input/r.png', 80, 350, 500);
    image('input/g.png', 83, 350, 500);
    image('input/b.png', 86, 350, 500);
    //makeLayer(layer, type, filter, dx, dy, imageSize)
    makeLine(end + 115, end + 135, 350, 1);
    end = makeLayer(0, 'Conv', 32, end + 120, 350, 416);
    end2 = makeLayer(0, 'Filter', 32, end + 20, 160, 220);
    makeLine(end + 35, end + 45, 350, 2);
    end = makeLayer(1, 'Maxpool', 32, end + 55, 350, 208);
    end2 = makeLayer(0, 'Filter', 32, end + 0, 160, 220);

    makeLine(end + 25, end + 35, 350, 3);
    end = makeLayer(2, 'Conv', 64, end + 40, 350, 208);
    end2 = makeLayer(4, 'Filter', 64, end - 20, 160, 220);
    makeLine(end + 20, end + 30, 350, 4);
    end = makeLayer(3, 'Maxpool', 64, end + 30, 350, 104);
    end2 = makeLayer(4, 'Filter', 64, end - 30, 160, 220);

    makeLine(end + 15, end + 25, 350, 5);
    end = makeLayer(4, 'Conv', 128, end + 30, 350, 104);
    end2 = makeLayer(8, 'Filter', 128, end - 60, 160, 220);
    makeLine(end + 15, end + 25, 350, 6);
    end = makeLayer(5, 'Conv', 64, end + 30, 350, 104);
    end2 = makeLayer(11, 'Filter', 64, end - 30, 160, 220);
    makeLine(end + 15, end + 25, 350, 7);
    end = makeLayer(6, 'Conv', 128, end + 30, 350, 104);
    end2 = makeLayer(8, 'Filter', 128, end - 60, 160, 220);
    makeLine(end + 15, end + 25, 350, 8);
    end = makeLayer(7, 'Maxpool', 128, end + 30, 350, 52);
    end2 = makeLayer(14, 'Filter', 128, end - 70, 160, 220);

    makeLine(end + 15, end + 25, 350, 9);
    end = makeLayer(8, 'Conv', 256, end + 30, 350, 52);
    end2 = makeLayer(18, 'Filter', 256, end - 130, 160, 220);
    makeLine(end + 10, end + 25, 350, 10);
    end = makeLayer(9, 'Conv', 128, end + 30, 350, 52);
    end2 = makeLayer(21, 'Filter', 128, end - 60, 160, 220);
    makeLine(end + 10, end + 25, 350, 11);
    end = makeLayer(10, 'Conv', 256, end + 30, 350, 52);
    end2 = makeLayer(18, 'Filter', 256, end - 130, 160, 220);
    makeLine(end + 10, end + 25, 350, 12);
    end = makeLayer(11, 'Maxpool', 256, end + 30, 350, 26);
    end2 = makeLayer(24, 'Filter', 256, end - 130, 160, 220);

    makeLine(end + 10, end + 25, 350, 13);
    end = makeLayer(12, 'Conv', 512, end + 30, 350, 26);
    end2 = makeLayer(28, 'Filter', 512, end - 260, 160, 220);
    makeLine(end + 10, end + 25, 350, 13);
    end = makeLayer(13, 'Conv', 256, end + 30, 350, 26);
    end2 = makeLayer(31, 'Filter', 256, end - 130, 160, 220);
    makeLine(end + 10, end + 25, 350, 14);
    end = makeLayer(14, 'Conv', 512, end + 30, 350, 26);
    end2 = makeLayer(34, 'Filter', 512, end - 260, 160, 220);
    makeLine(end + 10, end + 25, 350, 15);
    end = makeLayer(15, 'Conv', 256, end + 30, 350, 26);
    end2 = makeLayer(37, 'Filter', 256, end - 130, 160, 220);
    makeLine(end + 10, end + 25, 350, 16);
    end = makeLayer(16, 'Conv', 512, end + 30, 350, 26);
    end2 = makeLayer(34, 'Filter', 512, end - 260, 160, 220);
    makeLine(end + 10, end + 25, 350, 17);
    end = makeLayer(17, 'Maxpool', 512, end + 30, 350, 13); //seperate
    end2 = makeLayer(40, 'Filter', 512, end - 260, 160, 220);

    makeCurveLine(end + 10, 360, 2800, 480, 480, 61)
    makeLine(end + 10, end + 35, 350, 18);
    end = makeLayer(18, 'Conv', 1024, end + 45, 350, 13);
    end2 = makeLayer(44, 'Filter', 1024, end - 520, 160, 220);
    makeLine(end + 10, end + 25, 350, 19);
    end = makeLayer(19, 'Conv', 512, end + 30, 350, 13);
    end2 = makeLayer(47, 'Filter', 512, end - 260, 160, 220);
    // makeCurveLine(end+10, 350, 30, 350, 290, 20)


    // end = 0;
    makeLine(end + 10, end + 25, 350, 60);
    end = makeLayer(20, 'Conv', 1024, end + 30, 350, 13);
    end2 = makeLayer(50, 'Filter', 1024, end - 520, 160, 220);
    makeLine(end + 10, end + 25, 350, 20);
    end = makeLayer(21, 'Conv', 512, end + 30, 350, 13);
    end2 = makeLayer(53, 'Filter', 512, end - 260, 160, 220);
    makeLine(end + 10, end + 25, 350, 21);
    end = makeLayer(22, 'Conv', 1024, end + 30, 350, 13);
    end2 = makeLayer(56, 'Filter', 1024, end - 520, 160, 220);
    makeLine(end + 10, end + 25, 350, 22);
    end = makeLayer(23, 'Conv', 1024, end + 30, 350, 13);
    end2 = makeLayer(59, 'Filter', 1024, end - 520, 160, 220);
    makeLine(end + 10, end + 25, 350, 23);
    end = makeLayer(24, 'Conv', 1024, end + 30, 350, 13);
    end2 = makeLayer(62, 'Filter', 1024, end - 520, 160, 220);
    // makeLine(end+10, end+25, 350, 24);
    // makeCurveLine(end+10, 350, 6000, 350, 200, 62)

    end = 2810;
    end = makeLayer(26, 'Conv', 64, end + 10, 480, 26);
    end2 = makeLayer(40, 'Filter', 64, end - 50, 600, 220);
    // makeLine(end+10, end+25, 180, 26);
    // end = makeLayer(27, 'Conv', 256, end+30, 180, 13);  // layer16 size/2 ''reorg'' 
    // end2 = makeLayer(27, 'Filter', 256, end-130, 160, 220);
    // makeCurveLine(end+10, 180, 5970, 240, 180, 62)
    makeCurveLine(end + 10, 480, 6050, 360, 480, 62)
    // makeLine(end+10, end+35, 160, 27); 

    // end = makeLayer(28, 'route', 1024, end+30, 1350, 26);  // route layer24 and layer26
    end = 6020;
    makeLine(end + 10, end + 35, 350, 27);
    end = makeLayer(29, 'Conv', 1280, end + 45, 350, 13);
    end2 = makeLayer(65, 'Filter', 1280, end - 660, 160, 220);
    makeLine(end + 10, end + 25, 350, 28);
    end = makeLayer(30, 'Conv', 425, end + 30, 350, 13);
    end2 = makeLayer(65, 'Filter', 425, end - 225, 160, 220);
    makeLine(end + 10, end + 25, 350, 29);

    image('predictions.jpg', end + 40, 350, 500);

  }

  //----- create button -----------
  // var button = d3.select('body')
  //   .append('button')
  //   .attr('id', 'button')
  //   .text("reset")
  //   .attr("style", "position:absolute; left:10px; top:10px;")

  // d3.select('#button').on('click', darknet_19);

  darknet_19();
  imageHover();
  // imageClick();



};