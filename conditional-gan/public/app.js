$(document).ready(function() {

  function updateImage(id){
    setInterval(function() {
      $("#img" + id).prop("src", "./result/" + id + ".jpg?" + +new Date())
    }, 100)
  }

  $(".generate").click(function(e){
    e.preventDefault()

    $.ajax({
      method: "get",
      url: "./generate",
      data: {
        hair: $(".hair").val(),
        eye: $(".eye").val(),
      },
      success: function(data) {
        updateImage(0)
        updateImage(1)
        updateImage(2)
        updateImage(3)
        updateImage(4)
        updateImage(5)
        updateImage(6)
        updateImage(7)
        updateImage(8)
        updateImage(9)
      }
    })
  })
})

