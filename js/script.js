function showExamples(exampleId1, tabElement) {
  var examples = document.getElementsByClassName('example');
  var tabs = document.getElementsByClassName('tab');

  // 隐藏所有的example容器
  for (var i = 0; i < examples.length; i++) {
    examples[i].style.display = 'none';
  }

  // 移除所有标签的active类
  for (var j = 0; j < tabs.length; j++) {
    tabs[j].classList.remove('active');
  }

  // 显示指定的example容器
  document.getElementById(exampleId1).style.display = 'block';

  // 如果提供了tabElement，则为其添加active类
  if (tabElement) {
    tabElement.classList.add('active');
  }
}

function select_one_scene(scene_id) {
  var scenes = document.getElementsByClassName('scene');

  // 隐藏所有的example容器
  for (var i = 0; i < scenes.length; i++) {
    console.log(scenes[i])
    scenes[i].style.display = 'none';
  }

  // 显示指定的example容器
  console.log(scene_id, document.getElementById(scene_id))
  document.getElementById(scene_id).style.display = 'block';
  var top_down_name = scene_id + "_top_down_tab";
  var tab_top_down = document.getElementById(top_down_name);
  showExamples(scene_id + "_top_down_image", tab_top_down);
}

function select_one_scene_view1(scene_id) {
  var scenes = document.getElementsByClassName('scene');

  // 隐藏所有的example容器
  for (var i = 0; i < scenes.length; i++) {
    console.log(scenes[i])
    scenes[i].style.display = 'none';
  }

  // 显示指定的example容器
  console.log(scene_id, document.getElementById(scene_id))
  document.getElementById(scene_id).style.display = 'block';
  var top_down_name = scene_id + "_view1_tab";
  var tab_top_down = document.getElementById(top_down_name);
  showExamples(scene_id + "_view1", tab_top_down);
}

function scpp_select_one_scene(scene_id) {
  var scenes = document.getElementsByClassName('scpp_scene');

  // 隐藏所有的example容器
  for (var i = 0; i < scenes.length; i++) {
    console.log(scenes[i])
    scenes[i].style.display = 'none';
  }
  // 显示指定的example容器
  console.log(scene_id, document.getElementById(scene_id))
  document.getElementById(scene_id).style.display = 'block';
  var view1_name = scene_id + "_view1_tab";
  var tab_view1 = document.getElementById(view1_name);
  showExamples(scene_id + "_view1", tab_view1);
}

function sc_select_one_scene(scene_id) {
  var scenes = document.getElementsByClassName('scene');

  // 隐藏所有的example容器
  for (var i = 0; i < scenes.length; i++) {
    console.log(scenes[i])
    scenes[i].style.display = 'none';
  }
  // 显示指定的example容器
  console.log(scene_id, document.getElementById(scene_id))
  document.getElementById(scene_id).style.display = 'block';
  var view1_name = scene_id + "_view1_tab";
  var tab_view1 = document.getElementById(view1_name);
  showExamples(scene_id + "_view1", tab_view1);
}

function slide_left() {
slider_window = document.getElementById('thumbnails-scroll');
slider_window.scrollLeft = 0;
}

function slide_right() {
slider_window = document.getElementById('thumbnails-scroll');
slider_window.scrollLeft += 1000;
}


// 当文档加载完成时，自动显示并激活Tab 2，并初始化BeforeAfter
document.addEventListener('DOMContentLoaded', function() {
  select_one_scene_view1("scannetpp_13c3");
});
