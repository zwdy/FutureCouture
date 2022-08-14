$(document).ready(function(){
    $("#but_upload").click(function(){
        var fd = new FormData();
        var files = $('#file')[0].files[0];
        fd.append('file', files);
        console.log(files)
        $.ajax({
            url: 'http://localhost:8080/images',
            type: 'post',
            data: fd,
            contentType: false,
            processData: false,
            success: function(response){
                if (response != 0) {
                    //console.log(response);
                    var urlstring = "http://localhost:8080/images" + "/" + response.image_id + ".png";
                    $("#previewimg").attr("src", urlstring);
                    $("#previewimg").show(); // Display image element
                }else{
                    alert('file not uploaded');
                }
            },
        });
    });
});

function uploadImage(styleId){
  var fd = new FormData();
        var files = $('#file')[0].files[0];
        fd.append('file', files);
        console.log(files)
        const url = 'http://localhost:8080/submit?styleid='+styleId
        console.log(url)
        $.ajax({
            url: url,
            type: 'post',
            data: fd,
            contentType: false,
            processData: false,
            success: function(response){
                if (response != 0) {
                    console.log(response);
                    var urlstring = "/images" + "/" + response.folder_id + "/generated/out_0.png";
                    console.log(urlstring)
                    $("#result_image").attr("src", urlstring);
                    //$("#previewimg").show(); // Display image element
                }else{
                    alert('file not uploaded');
                }
            },
        });}

  function imageChanged(e){
  var file = $('#file')[0].files[0];
const reader = new FileReader()
reader.onload= () => {

    $("#previewimg").attr("src", reader.result);
    }
const data = reader.readAsDataURL(file)
console.log("here")
  }
