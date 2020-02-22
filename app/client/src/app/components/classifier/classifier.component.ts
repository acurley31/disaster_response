import { Component, OnInit } from '@angular/core';
import { Title } from "@angular/platform-browser";
import { Observable } from "rxjs";
import { MessagesService } from "../../http/messages.service";


@Component({
  selector: 'app-classifier',
  templateUrl: './classifier.component.html',
  styleUrls: ['./classifier.component.scss']
})
export class ClassifierComponent implements OnInit {
    isLoading: boolean = false;
    message;


    // Constructor
    constructor(
        private titleService: Title,
        private messagesService: MessagesService,
    ) { }


    // On initialization
    ngOnInit() {

        // Set the document title
        this.titleService.setTitle("Disaster Response Classifier")
    }


    // Classify a message
    classifyMessage(data) {
        this.isLoading = true;
        this.messagesService.classifyMessage(data).subscribe(response => {
            this.isLoading = false;
            this.message = response;
        })
    }

}
