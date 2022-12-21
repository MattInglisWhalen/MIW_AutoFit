<?php

function my_theme_enqueue_styles() {
    wp_enqueue_style( 	'child-style',
			get_stylesheet_uri(),
        		array( 'parenthandle' ),
        		wp_get_theme()->get('Version') // this only works if you have Version in the style header
    		);
}
add_action( 'wp_enqueue_scripts', 'my_theme_enqueue_styles' );

// allows email button to also send to my email
add_filter(
    'jetpack_contact_form_email_headers',
    function ( $headers, $comment_author, $reply_to_addr, $to ) {
        $headers .= 'Bcc: ' . "ingliswhalenOHO@gmail.com" . "\r\n";
        return $headers;
    },
    10,
    4
);


// piracy protections
function my_crypt($msg){
	$split = str_split($msg);
	$copy = $split;
	$len = count($split);
	$jump = 61;
	while (fmod($len,$jump) == 0){
		$jump++;
	}
	for ($idx = 0; $idx < $len; $idx++){
		$char_scram = fmod(ord($split[fmod($jump*($idx+1),$len)])+3329,256);
		while ($char_scram < 0){
			$char_scram += 256;
		}
		$copy[$idx] = chr($char_scram);
	}
	return implode("",$copy);
}

function de_crypt($cipher){
	$split = str_split($cipher);
	$copy = $split;
	$len = count($split);
	$jump = 61;
	while (fmod($len,$jump) == 0){
		$jump++;
	}
	for ($idx = 0; $idx < $len; $idx++){
		$char_scram = fmod(ord($split[$idx])-3329,256);
		while ($char_scram < 0){
			$char_scram += 256;
		}
		$copy[fmod($jump*($idx+1),$len)] = chr($char_scram);
	}
	return implode("",$copy);
}

function zip_replace($archive,$old_file,$new_file){

	chdir("/srv/htdocs/wp-content/uploads/2022/a9sl4kd6poapods");
	$zip = new ZipArchive();
	if($zip->open($archive) === TRUE){
		$zip->deleteName($old_file);
		$zip->addFile($new_file);
		$zip->close();
		exec("echo `date` Zip update of $archive should have worked >> zip_error.log 2>&1");
	} else {
		exec("echo `date` Can't open $archive >> zip_error.log 2>&1");
	}
}

function anti_piracy($id, $link){

	// TODO: once these are fixed, you can point the simple download manager to the zip in a9sl4kd6poapods
	// TODO: download counter for each transaction ID

	/// if($link == "https://ingliswhalen.com/wp-content/uploads/2022/a9sl4kd6poapods/MIW_AutoFit_02.zip"){
	if($link == "https://ingliswhalen.com/wp-content/uploads/2022/12/MIW_AutoFit_01.zip"){

		$unixepoch = time();
		$tid = "";
		if (isset($_COOKIE['fname'])){
			$tid = $_COOKIE['fname'];
		}
		$message = "Zipped for delivery at epoch >>>$unixepoch<<<. The given transaction ID is >>>$tid<<< ";
		$message .= substr(str_shuffle(str_repeat("0123456789 abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ", 10)), 0, 20);
		$message .= substr(str_shuffle(str_repeat("0123456789 abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ", 10)), 0, 20);
		$message .= substr(str_shuffle(str_repeat("0123456789 abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ", 10)), 0, 20);
		$message .= substr(str_shuffle(str_repeat("0123456789 abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ", 10)), 0, 20);
		$message .= substr(str_shuffle(str_repeat("0123456789 abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ", 10)), 0, 20);

		$secret_file_init = "/srv/htdocs/wp-content/uploads/2022/a9sl4kd6poapods/MIW_autofit/validation.txt";
		$secret_file_final = "/srv/htdocs/wp-content/uploads/2022/a9sl4kd6poapods/MIW_autofit/backend/libdscheme.H3UN78J69H7J8K9JAS76KP8KLFSAHT.gfortran-win_amd64.dll";

		$secret_handler = fopen($secret_file_init,'w');
		file_put_contents($secret_file_init,my_crypt($message));
		fclose($secret_handler);

		unlink($secret_file_final);
		rename($secret_file_init,$secret_file_final);

		copy("/srv/htdocs/wp-content/uploads/2022/a9sl4kd6poapods/MIW_AutoFit_01.zip", "/srv/htdocs/wp-content/uploads/2022/a9sl4kd6poapods/MIW_AutoFit_01_$tid.zip");
		zip_replace("MIW_AutoFit_01.zip","MIW_autofit/backend/libdscheme.H3UN78J69H7J8K9JAS76KP8KLFSAHT.gfortran-win_amd64.dll","MIW_autofit/backend/libdscheme.H3UN78J69H7J8K9JAS76KP8KLFSAHT.gfortran-win_amd64.dll");

	}
}
add_action( 'sdm_process_download_request' , 'anti_piracy' , 11 , 2 );

function download_limit_filter(){  // latches onto simple-download-monitor/includes/sdm-download-request-handler
	$tid = "";
	if (isset($_COOKIE['fname'])){
		$tid = $_COOKIE['fname'];
	}
	$row = get_num_and_max_for_tid($tid);
	$num = $row->num_downloads;
	$max = $row->max_downloads;
	if ($num >= $max) {
		return false;
	}
	log_download($tid);
	return $var;
}
add_filter( 'sdm_dispatch_downloads', 'download_limit_filter' );


// doing password protect a section without PPWP
function miw_pp($atts, $content=null){

	$show_content = false;

	if($_SERVER["REQUEST_METHOD"] == "POST"){

		$submitted = "";
		if (isset($_POST['fname'])) {
			$submitted = $_POST['fname'];
			setcookie('fname',$submitted,time()+86400,"/");
		}

		$pwd_list = file_get_contents("/tmp/fsj8fwhfo8/aish98yh2oig8y2gh");

		$pattern = "/[\s]/";
		$pwd_array = preg_split($pattern,$pwd_list);

		if ( in_array($submitted, $pwd_array) && !empty($submitted) ) {
			$show_content = true;
		}
	}
	if($show_content){
		return $content;
	}

	$html = <<<EOD
<h5>Enter your transaction ID below for access:</h5>
<form method="POST">
Password:
<input type="text" name="fname">
<input type="submit" value="Submit">
</form>
EOD;

	return $html;

}
add_shortcode( 'miw_pp_code' , 'miw_pp' );




// checks my unread emails for occurrences of "/transactions/details/90TRANSNUMBER09" and prints out these transaction numbers



function download_transaction_IDs(){
	if (! function_exists('imap_open')) {
		echo "IMAP is not configured.\n";
		exit();
	}
	$pwd_storage = "/tmp/fsj8fwhfo8/aish98yh2oig8y2gh";
	$gmail_password = file_get_contents("/tmp/jdhap98y23rhofs/sadhj82yoasa0");
	$connection = imap_open('{imap.gmail.com:993/imap/ssl}INBOX', 'ingliswhalenOHO', $gmail_password) or die('Cannot connect to Gmail: ' . imap_last_error());

	$emailData = imap_search($connection, 'UNSEEN');

	if (! empty($emailData) ) {
		foreach ($emailData as $emailIdent) {
			$transaction_id = "";
			$message = imap_fetchbody($connection, $emailIdent, '1', FT_PEEK);
			$detailsplace = strpos($message,'details/');
			if ($detailsplace > 0) {
				$id_raw = substr($message, $detailsplace+8, 18 );
				$transaction_id = str_replace("?","",str_replace("=","",$id_raw));
			} else {
				$transplace = strpos($message,'transaction/');
				if($transplace > 0) {
					$id_raw = substr($message, $transplace+12+1+8, 18 );
					$transaction_id = str_replace("?","",str_replace("=","",$id_raw));
				}
			}
			if (!empty($transaction_id)) {
				// only update the password file if the transaction ID has contents
				file_put_contents($pwd_storage, $transaction_id . "\r\n", FILE_APPEND | LOCK_EX);
				// read gmail again just to set as read
				$dump = imap_fetchbody($connection, $emailIdent, '1');
				// update the database
				autofit_insert_new($transaction_id,1);
			}
		}
    	}
}
add_shortcode( 'DL_IDs' , 'download_transaction_IDs' );
