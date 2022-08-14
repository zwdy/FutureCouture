
let camera_button = document.querySelector("#start-camera");
let video = document.querySelector("#video");
let click_button = document.querySelector("#click-photo");
let canvas = document.querySelector("#canvas");
let dataurl = document.querySelector("#dataurl");
let dataurl_container = document.querySelector("#dataurl-container");

camera_button.addEventListener('click', async function() {
    $("#previewimg").hide()
    let stream = null;

    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    }
    catch(error) {
      alert(error.message);
      return;
    }

    video.srcObject = stream;

    video.style.display = 'block';
    camera_button.style.display = 'none';
    click_button.style.display = 'block';
});

click_button.addEventListener('click', function(styleId) {

    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    let image_data_url = canvas.toDataURL('image/jpeg');

    var dataURL = canvas.toDataURL('image/jpeg', 0.5);
    var blob = dataURItoBlob(dataURL);
    var files = new File( [blob], 'canvasImage.jpg', { type: 'image/jpeg' } );

    const dataTransfer = new DataTransfer()
    dataTransfer.items.add(files)
    $('#file')[0].files = dataTransfer.files
    $("#previewimg").attr("src", dataURL);

    $("#video").hide()
    $("#previewimg").show()


     // var file = $('#file')[0].files = [files];
    console.log("add file")

    return;

    var dataURL = canvas.toDataURL('image/jpeg', 0.5);
    var blob = dataURItoBlob(dataURL);
    var fd = new FormData(document.forms[0]);
    var files = new File( [blob], 'canvasImage.jpg', { type: 'image/jpeg' } );
    fd.append('file', files);
    console.log(files)
    const url = 'http://localhost:8080/submit?styleid=style01.png' //?? need to copy after click another button
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
            }else{
                alert('file not uploaded');
            }
        },
    });
});

function dataURItoBlob(dataURI) {
    // convert base64/URLEncoded data component to raw binary data held in a string
    var byteString;
    if (dataURI.split(',')[0].indexOf('base64') >= 0)
        byteString = atob(dataURI.split(',')[1]);
    else
        byteString = unescape(dataURI.split(',')[1]);

    // separate out the mime component
    var mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0];

    // write the bytes of the string to a typed array
    var ia = new Uint8Array(byteString.length);
    for (var i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i);
    }

    return new Blob([ia], {type:mimeString});
}


